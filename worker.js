// OC 心理侧写与 CP 匹配引擎 —— Worker 单文件版
// 绑定：KV namespace → OC_KV ｜ 变量：DEEPSEEK_API_KEY、SERVER_CHAN_KEY
// 测试版：不限额 / 不扣费；先指定角色名再锁定主人公；请求体上限 512KB

/* ================= worker/kv.js ================= */
// worker/kv.js
// KV 数据层：OC_KV 绑定的最小读写封装（JSON 序列化 + 用户记录命名空间）

const USER_KEY_PREFIX = 'user:';

/**
 * 取出并校验 KV 绑定，缺失时抛出可被入口识别的错误（带 code / status）
 */
function binding(env) {
  const kv = env && env.OC_KV;
  if (!kv || typeof kv.get !== 'function' || typeof kv.put !== 'function') {
    const error = new Error('KV 绑定 "OC_KV" 未配置，请检查 Worker 的 KV Namespace 绑定');
    error.code = 'KV_BINDING_MISSING';
    error.status = 500;
    throw error;
  }
  return kv;
}

/**
 * 用户记录的存储键：user:KEY-XXXX-XXXX
 */
function userStorageKey(key) {
  return `${USER_KEY_PREFIX}${String(key).trim().toUpperCase()}`;
}

/**
 * 基础读取
 * @returns {any|null} 反序列化后的值；不存在或脏数据返回 null
 */
async function kvGet(env, key, { raw = false } = {}) {
  if (!key) return null;
  const stored = await binding(env).get(key);
  if (stored === null || stored === undefined || stored === '') return null;
  if (raw) return stored;
  try {
    return JSON.parse(stored);
  } catch {
    console.warn('[kv] 脏数据，按 null 处理:', key);
    return null;
  }
}

/**
 * 基础写入（对象自动 JSON 序列化）
 * @param {object} [options]
 * @param {number} [options.expirationTtl] 秒级过期，KV 最小 60 秒，不传则永久
 * @param {object} [options.metadata] 列表查询用的元数据
 */
async function kvPut(env, key, value, { expirationTtl, metadata } = {}) {
  if (!key) {
    const error = new Error('KV key 不能为空');
    error.code = 'KV_INVALID_KEY';
    error.status = 500;
    throw error;
  }

  const payload = typeof value === 'string' ? value : JSON.stringify(value);
  if (payload === undefined) {
    const error = new Error('KV value 无法序列化');
    error.code = 'KV_INVALID_VALUE';
    error.status = 500;
    throw error;
  }

  const options = {};
  if (Number.isInteger(expirationTtl) && expirationTtl >= 60) options.expirationTtl = expirationTtl;
  if (metadata !== undefined) options.metadata = metadata;

  await binding(env).put(key, payload, Object.keys(options).length ? options : undefined);
  return true;
}

/* ================= 用户记录 ================= */

async function getUser(env, key) {
  return kvGet(env, userStorageKey(key));
}

async function putUser(env, key, user) {
  return kvPut(env, userStorageKey(key), user);
}


/* ================= worker/deepseek.js ================= */
// worker/deepseek.js
// DeepSeek API 代理：动态模型路由 + 超时控制 + 指数退避重试 + JSON 载荷提取

const API_ENDPOINT = 'https://api.deepseek.com/chat/completions';

const MODEL_CHAT = 'deepseek-chat';
const MODEL_REASONER = 'deepseek-reasoner';

// 字符数阈值：小于等于 8000 用 chat，大于 8000 用 reasoner
const LONG_TEXT_THRESHOLD = 8000;

const MAX_ATTEMPTS = 3;
const RETRYABLE_STATUS = new Set([408, 409, 429, 500, 502, 503, 504]);

// 不同模型的等待预算（毫秒），reasoner 需要更长的推理时间
const MODEL_TIMEOUT_MS = {
  [MODEL_CHAT]: 60000,
  [MODEL_REASONER]: 150000,
};

function deepseekError(code, message, status = 502, detail) {
  const error = new Error(message);
  error.code = code;
  error.status = status;
  if (detail) error.detail = detail;
  return error;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * 动态路由：文本越长越需要推理，切换到 deepseek-reasoner
 */
function pickModel(text) {
  const length = typeof text === 'string' ? text.length : 0;
  return length > LONG_TEXT_THRESHOLD ? MODEL_REASONER : MODEL_CHAT;
}

async function fetchWithTimeout(url, init, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (error) {
    if (error && error.name === 'AbortError') {
      const timeoutError = deepseekError('DEEPSEEK_TIMEOUT', 'DeepSeek 请求超时，请缩短文本或稍后重试', 504);
      timeoutError.retryable = false;
      throw timeoutError;
    }
    const networkError = deepseekError('DEEPSEEK_NETWORK_ERROR', 'DeepSeek 网络请求失败', 502);
    networkError.retryable = true;
    throw networkError;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * 调用 DeepSeek Chat Completions
 * @returns {{ model:string, content:string, reasoning:string, usage:object|null, latencyMs:number, attempts:number }}
 */
async function callDeepSeek(env, options = {}) {
  const {
    system,
    user,
    model = MODEL_CHAT,
    json = false,
    temperature = 0.7,
    maxTokens = 1024,
    timeoutMs,
    retries = MAX_ATTEMPTS,
  } = options;

  const apiKey = env && env.DEEPSEEK_API_KEY;
  if (!apiKey) throw deepseekError('DEEPSEEK_KEY_MISSING', 'DeepSeek API Key 未配置', 500);
  if (typeof user !== 'string' || !user) {
    throw deepseekError('DEEPSEEK_EMPTY_INPUT', '待分析文本为空', 400);
  }

  const messages = [];
  if (system) messages.push({ role: 'system', content: system });
  messages.push({ role: 'user', content: user });

  const payload = { model, messages, stream: false };
  // reasoner 的 max_tokens 与思维链共处一个预算池，需要留足余量
  payload.max_tokens = model === MODEL_REASONER ? Math.max(maxTokens, 8192) : maxTokens;

  if (model === MODEL_CHAT) {
    payload.temperature = temperature;
    // JSON 模式仅 chat 支持；reasoner 靠提示词约束 + 本地提取兜底
    if (json) payload.response_format = { type: 'json_object' };
  }

  const budget = Number.isFinite(timeoutMs) ? timeoutMs : MODEL_TIMEOUT_MS[model] || 60000;
  const startedAt = Date.now();
  const totalAttempts = Math.max(1, retries);

  let lastError = null;

  for (let attempt = 1; attempt <= totalAttempts; attempt += 1) {
    try {
      const response = await fetchWithTimeout(
        API_ENDPOINT,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
            Authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify(payload),
        },
        budget
      );

      if (!response.ok) {
        const raw = await response.text().catch(() => '');
        const error = deepseekError(
          'DEEPSEEK_HTTP_ERROR',
          `DeepSeek 接口返回 ${response.status}`,
          response.status === 429 ? 429 : 502,
          raw.slice(0, 300)
        );
        error.retryable = RETRYABLE_STATUS.has(response.status);
        error.upstreamStatus = response.status;
        throw error;
      }

      const data = await response.json();
      const choice = data && data.choices && data.choices[0];
      const content = (choice && choice.message && choice.message.content) || '';
      const reasoning = (choice && choice.message && choice.message.reasoning_content) || '';

      const finishReason = (choice && choice.finish_reason) || 'unknown';

      if (!content.trim()) {
        // 空回复：模型把预算耗在思维链上，或被安全策略清空
        const error = deepseekError(
          'DEEPSEEK_EMPTY_CONTENT',
          `DeepSeek 返回内容为空（finish_reason=${finishReason}，思维链 ${reasoning.length} 字）`,
          502
        );
        error.retryable = true;
        error.finishReason = finishReason;
        throw error;
      }

      return {
        model: (data && data.model) || model,
        content,
        reasoning,
        finishReason,
        usage: data && data.usage ? data.usage : null,
        latencyMs: Date.now() - startedAt,
        attempts: attempt,
      };
    } catch (error) {
      lastError = error;
      const retryable = error && error.retryable === true;
      if (!retryable || attempt >= totalAttempts) break;
      // 指数退避 + 抖动：300ms / 600ms / 1200ms
      await sleep(300 * 2 ** (attempt - 1) + Math.floor(Math.random() * 200));
    }
  }

  throw lastError || deepseekError('DEEPSEEK_UNKNOWN', 'DeepSeek 调用失败', 502);
}

/**
 * 从模型输出里提取 JSON：兼容代码围栏、前后废话、尾随逗号
 */
function extractJson(content) {
  if (typeof content !== 'string') return null;

  let text = content.trim();
  const fenced = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (fenced) text = fenced[1].trim();

  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) return null;

  const candidate = text.slice(start, end + 1);

  try {
    return JSON.parse(candidate);
  } catch {
    try {
      return JSON.parse(candidate.replace(/,\s*([}\]])/g, '$1'));
    } catch {
      return null;
    }
  }
}


/* ================= worker/prompts.js ================= */
// worker/prompts.js
// System Prompt：OC 角色 -> 10 维评分 + 读者视角评价 + 维度解读

// 10 个维度键（英文）与中文含义一一对应
const DIMENSION_KEYS = [
  'empathy',          // 共情力：理解并回应他人情绪的能力
  'order',            // 秩序感：对规则、结构、可预测性的需求强度
  'intellect',        // 智谋度：推理、布局、操纵局势的能力
  'vitality',         // 向阳心：生命力、乐观与自我修复倾向
  'pride',            // 自尊感：对自我价值与尊严的维护强度
  'dominance',        // 支配欲：控制他人与局面的欲望
  'self_destruction', // 自毁度：主动消耗、伤害自身的倾向
  'masochism',        // 痛觉度：对痛苦、牺牲、受虐的耐受与沉迷
  'obsession',        // 偏执狂：执念、占有欲、思维固着的强度
  'destruction',      // 破坏欲：摧毁外物与他人世界秩序的攻击性
];

const DIMENSION_LABELS = {
  empathy: '共情力',
  order: '秩序感',
  intellect: '智谋度',
  vitality: '向阳心',
  pride: '自尊感',
  dominance: '支配欲',
  self_destruction: '自毁度',
  masochism: '痛觉度',
  obsession: '偏执狂',
  destruction: '破坏欲',
};

const PARSE_SYSTEM_PROMPT = `你是「OC 心理侧写引擎」的核心分析器，服务对象是角色扮演与同人创作的作者。
你的任务：在给定文本中锁定用户指定的角色，输出 ta 的 10 维心理评分、一段读者视角的评价、以及逐维度解读。

## 评分维度（0-100 的整数，50 = 普通成年人的中位水平）
1. empathy 共情力：0 = 读不懂他人情绪、视人如物；100 = 他人之痛如切身之痛。
2. order 秩序感：0 = 厌恶规则、随性混乱；100 = 强迫性秩序。
3. intellect 智谋度：0 = 单纯直给；100 = 多线布局、算无遗策。
4. vitality 向阳心：0 = 彻底厌世；100 = 在废墟里也能长出花。
5. pride 自尊感：0 = 任人践踏；100 = 宁折不弯。
6. dominance 支配欲：0 = 甘居从属；100 = 必须掌控一切。
7. self_destruction 自毁度：0 = 极度自保；100 = 主动走向毁灭。
8. masochism 痛觉度：0 = 对痛零耐受；100 = 依赖疼痛、把受苦当作存在证明。
9. obsession 偏执狂：0 = 随遇而安；100 = 单一执念吞噬一切。
10. destruction 破坏欲：0 = 维护秩序；100 = 以摧毁他人与世界为乐。

## 角色锁定
- 用户会指定目标角色名。文本里常有多个角色，只评估 target 本人的言行与心理，绝不把别人的特质算给 ta。
- 全名 / 简称 / 称号 / 昵称视为同一角色。
- 完全找不到该角色（或只有名字、无可依据内容）时，只返回 {"error":"CHARACTER_NOT_FOUND"}。

## 批注（review）怎么写
- 你是刚读完这段文本的人，心里有话想找人说说，语气自然、具体、克制。
- 只夸这个角色，用不会出错的词：认真、体面、倔、清醒、柔软、有种、拎得清、心气高。
- 落到文本里的具体动作或一句话上，再轻轻收住，不要下结论式的判词。
- 禁止句式：「不是……而是……」「不仅……更是……」「与其说……不如说……」这类对仗；禁止「让我最被打动的是」「作为读者」「不难看出」这类开场；禁止三段式排比和升华式结尾。
- 不要复述剧情大段，不要出现 AI 腔（如"不禁让人思考""展现了人性的复杂"）。
- 篇幅：120-200 字，一段话，中文。

## 维度解读（dimensions）
- 10 个键各写一句 15-40 字的解读，说清"为什么给这个分"，尽量落到文本细节。
- 不要重复分数本身，不要写成词条堆砌。

## 输出格式（严格遵守）
只输出一个 JSON 对象，结构如下，不要任何解释、前后缀或 markdown 代码块：
{"scores":{"empathy":0,"order":0,"intellect":0,"vitality":0,"pride":0,"dominance":0,"self_destruction":0,"masochism":0,"obsession":0,"destruction":0},"review":"……","dimensions":{"empathy":"……","order":"……","intellect":"……","vitality":"……","pride":"……","dominance":"……","self_destruction":"……","masochism":"……","obsession":"……","destruction":"……"}}
上面的 0 与省略号只是占位，务必替换为真实内容。`;

function buildParseUserPrompt(text, characterName, mbti) {
  const lines = [
    `本次要侧写的目标角色是「${characterName}」。`,
  ];

  if (mbti) {
    lines.push(
      `该角色的设定 MBTI 为 ${mbti}。请将 10 维分析严格与 MBTI 的认知功能（如 Ni, Te, Fi 等）对齐。如果文本与 MBTI 有反差，请着重分析这种反差感带来的张力。`
    );
  }

  lines.push(
    '文本里可能同时出现多个角色，请只针对这个目标角色评分与撰写评价。',
    '如果文本里找不到这个角色（或只有名字出现、没有任何可依据的内容），只返回 {"error":"CHARACTER_NOT_FOUND"}。',
    '如果能找到，按系统提示的 JSON 结构返回。',
    '<oc_text>',
    text,
    '</oc_text>'
  );

  return lines.join('\n');
}


/* ================= worker/push.js ================= */
// worker/push.js
// Server酱推送封装：用于反馈通知，失败不影响主流程

const TIMEOUT_MS = 8000;

/**
 * 兼容两种 SendKey：
 * - sctpXXXX（Server酱³）：https://<SENDKEY>.push.ft07.com/send
 * - SCTXXXX（Turbo）：https://sctapi.ftqq.com/<SENDKEY>.send
 */
function endpoint(key) {
  return key.startsWith('sctp')
    ? `https://${key}.push.ft07.com/send`
    : `https://sctapi.ftqq.com/${key}.send`;
}

/**
 * 发送通知（永不抛错，只返回结果对象）
 * @param {object} env Worker 环境变量（读取 SERVER_CHAN_KEY）
 * @param {{ title?: string, content?: string }} message
 */
async function pushToServerChan(env, { title, content } = {}) {
  const sendKey = env && env.SERVER_CHAN_KEY;
  if (!sendKey) {
    console.warn('[push] SERVER_CHAN_KEY 未配置，跳过推送');
    return { ok: false, skipped: true, reason: 'SERVER_CHAN_KEY 未配置' };
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const body = new URLSearchParams({
      title: String(title || 'OC 心理侧写引擎通知').slice(0, 100),
      desp: String(content || ''),
    });

    const response = await fetch(endpoint(sendKey), {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: body.toString(),
      signal: controller.signal,
    });

    const raw = await response.text();
    let payload = null;
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = null;
    }

    // Server酱成功时 code 为 0
    const code = payload && typeof payload.code === 'number' ? payload.code : undefined;
    const ok = response.ok && (code === 0 || code === undefined);
    if (!ok) console.warn('[push] 推送失败', response.status, raw.slice(0, 200));

    return { ok, status: response.status, code };
  } catch (error) {
    console.warn('[push] 推送异常', error && error.message);
    return { ok: false, error: (error && error.message) || 'push failed' };
  } finally {
    clearTimeout(timer);
  }
}


/* ================= worker/index.js ================= */
// worker/index.js
// OC 心理侧写与 CP 匹配引擎 —— Worker 入口
// 职责：CORS 放行 / 极简路由分发 / 灵魂契约签发（POST /api/auth）/ 登录（POST /api/login）






/* ================= 常量 ================= */

const KEY_PREFIX = 'KEY';
// 剔除易混淆字符 I、O、0、1 后恰好 32 个字符；256 可被 32 整除，取模无分布偏差
const KEY_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
const KEY_PATTERN = /^KEY-[A-HJ-NP-Z2-9]{4}-[A-HJ-NP-Z2-9]{4}$/;
const KEY_ISSUE_ATTEMPTS = 5;

const NICKNAME_MAX_LENGTH = 24;
// 中文 3 字节/字，64KB 只能装约 2.1 万字，放宽到 512KB 才配得上 3 万字上限
const MAX_BODY_BYTES = 512 * 1024;
const INITIAL_BALANCE = 5;

// 侧写计费参数
const TEXT_MAX_LENGTH = 30000; // 单次提交文本上限（字符）
const CHAR_UNIT = 1000; // 计费阶梯：每 1000 字
const CHARGE_PER_UNIT = 1; // 每个阶梯扣 1 点
const MIN_CHARGE = 1; // 单次最低扣费
const PARSE_TEMPERATURE = 0.6;
const PARSE_MAX_TOKENS = 1600; // 留足预算，避免 JSON 被截断成半截

// 测试期开关：true = 不检查余额、不扣费
const UNLIMITED_MODE = true;

// 反馈与券码
const FEEDBACK_MIN_LENGTH = 20;
const FEEDBACK_MAX_LENGTH = 200;
const FEEDBACK_MIN_MEANINGFUL = 8; // 至少 8 个字母/数字，过滤纯表情与纯符号
const COUPON_PREFIX = 'VIP';
const COUPON_PATTERN = /^VIP-[A-HJ-NP-Z2-9]{4}$/;
const COUPON_REWARD = 5; // 每张体验券兑换 5 虚拟币
const COUPON_ISSUE_ATTEMPTS = 8;

// 角色码：OC-XXXX，跨账号可导入
const OC_PREFIX = 'OC';
const OC_CODE_PATTERN = /^OC-[A-HJ-NP-Z2-9]{4}$/;
const OC_CODE_ATTEMPTS = 8;

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, PATCH, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  'Access-Control-Max-Age': '86400',
};

/* ================= 错误与响应 ================= */

class HttpError extends Error {
  constructor(status, code, message, headers) {
    super(message);
    this.name = 'HttpError';
    this.status = status;
    this.code = code;
    this.headers = headers;
  }
}

function jsonResponse(payload, status = 200, extraHeaders) {
  const headers = new Headers({
    ...CORS_HEADERS,
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  });
  if (extraHeaders) {
    for (const [name, value] of Object.entries(extraHeaders)) {
      if (value !== undefined && value !== null) headers.set(name, value);
    }
  }
  return new Response(JSON.stringify(payload), { status, headers });
}

// 统一成功信封：{ ok: true, data }
function ok(data, status = 200) {
  return jsonResponse({ ok: true, data }, status);
}

// 统一失败信封：{ ok: false, error: { code, message } }
function fail(error) {
  const known = typeof error?.code === 'string';
  const status = known && Number.isInteger(error.status) ? error.status : 500;
  return jsonResponse(
    {
      ok: false,
      error: {
        code: known ? error.code : 'INTERNAL_ERROR',
        message: known ? error.message : '服务内部错误，请稍后重试',
      },
    },
    status,
    known ? error.headers : undefined
  );
}

/* ================= 入参处理 ================= */

async function readJsonBody(request) {
  const declaredLength = Number(request.headers.get('content-length') || 0);
  if (declaredLength > MAX_BODY_BYTES) {
    throw new HttpError(413, 'PAYLOAD_TOO_LARGE', '请求体过大');
  }

  const text = await request.text();
  if (!text.trim()) throw new HttpError(400, 'EMPTY_BODY', '请求体不能为空');
  if (text.length > MAX_BODY_BYTES) throw new HttpError(413, 'PAYLOAD_TOO_LARGE', '请求体过大');

  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new HttpError(400, 'INVALID_JSON', '请求体不是合法 JSON');
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new HttpError(400, 'INVALID_BODY', '请求体必须是 JSON 对象');
  }
  return parsed;
}

function normalizeNickname(raw) {
  if (typeof raw !== 'string') throw new HttpError(400, 'INVALID_NICKNAME', '昵称必须为字符串');

  const cleaned = raw
    .replace(/[\u0000-\u001f\u007f]/g, '') // 去掉控制字符
    .replace(/\s+/g, ' ')
    .trim();

  if (!cleaned) throw new HttpError(400, 'INVALID_NICKNAME', '昵称不能为空');
  if ([...cleaned].length > NICKNAME_MAX_LENGTH) {
    throw new HttpError(400, 'INVALID_NICKNAME', `昵称不能超过 ${NICKNAME_MAX_LENGTH} 个字符`);
  }
  return cleaned;
}

function normalizeKey(raw) {
  if (typeof raw !== 'string' || !raw.trim()) {
    throw new HttpError(400, 'INVALID_KEY', '请输入灵魂契约密钥');
  }

  const compact = raw.trim().toUpperCase().replace(/[\s_]+/g, '-');
  const normalized = compact.startsWith(`${KEY_PREFIX}-`) ? compact : `${KEY_PREFIX}-${compact}`;

  if (!KEY_PATTERN.test(normalized)) {
    throw new HttpError(400, 'INVALID_KEY_FORMAT', '密钥格式应为 KEY-XXXX-XXXX');
  }
  return normalized;
}

/* ================= 密钥 ================= */

function randomKey() {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  const chars = Array.from(bytes, (byte) => KEY_ALPHABET[byte % KEY_ALPHABET.length]);
  return `${KEY_PREFIX}-${chars.slice(0, 4).join('')}-${chars.slice(4).join('')}`;
}

async function issueKey(env) {
  for (let attempt = 0; attempt < KEY_ISSUE_ATTEMPTS; attempt += 1) {
    const key = randomKey();
    const taken = await getUser(env, key);
    if (!taken) return key;
  }
  throw new HttpError(500, 'KEY_ISSUE_FAILED', '密钥签发失败，请重试');
}

/* ================= 角色档（OC） ================= */

function ocStorageKey(code) {
  return `oc:${String(code).trim().toUpperCase()}`;
}

function randomOcCode() {
  const bytes = new Uint8Array(4);
  crypto.getRandomValues(bytes);
  const chars = Array.from(bytes, (byte) => KEY_ALPHABET[byte % KEY_ALPHABET.length]);
  return `${OC_PREFIX}-${chars.join('')}`;
}

async function issueOcCode(env) {
  for (let attempt = 0; attempt < OC_CODE_ATTEMPTS; attempt += 1) {
    const code = randomOcCode();
    const taken = await kvGet(env, ocStorageKey(code));
    if (!taken) return code;
  }
  throw new HttpError(500, 'OC_CODE_FAILED', '角色码生成失败，请重试');
}

async function getOc(env, code) {
  return kvGet(env, ocStorageKey(code));
}

async function putOc(env, profile) {
  return kvPut(env, ocStorageKey(profile.code), profile);
}

function shortText(value, max) {
  const text = typeof value === 'string' ? value.trim() : '';
  return text.slice(0, max);
}

function sanitizeDimensions(raw) {
  const source = raw && typeof raw === 'object' ? raw : {};
  const dimensions = {};
  for (const dimension of DIMENSION_KEYS) {
    dimensions[dimension] = shortText(source[dimension], 80);
  }
  return dimensions;
}

/* ================= 用户数据 ================= */

function createUserRecord(key, nickname) {
  const now = new Date().toISOString();
  return {
    key,
    nickname,
    balance: INITIAL_BALANCE,
    coupons: [],
    oc_records: [],
    createdAt: now,
    lastLoginAt: now,
  };
}

// 兼容历史数据缺字段，保证前端拿到的结构恒定
function reviveUser(record) {
  const user = { ...record };
  if (typeof user.balance !== 'number' || !Number.isFinite(user.balance)) user.balance = INITIAL_BALANCE;
  if (!Array.isArray(user.coupons)) user.coupons = [];
  if (!Array.isArray(user.oc_records)) user.oc_records = [];
  if (typeof user.nickname !== 'string' || !user.nickname) user.nickname = '未命名';
  return user;
}

/* ================= 业务接口 ================= */

// POST /api/auth { nickname } -> { key, user }
async function handleAuth(request, env) {
  const body = await readJsonBody(request);
  const nickname = normalizeNickname(body.nickname);

  const key = await issueKey(env);
  const user = createUserRecord(key, nickname);
  await putUser(env, key, user);

  return { key, user };
}

// POST /api/login { key } -> { key, user }
async function handleLogin(request, env) {
  const body = await readJsonBody(request);
  const key = normalizeKey(body.key);

  const record = await getUser(env, key);
  if (!record) throw new HttpError(404, 'USER_NOT_FOUND', '灵魂契约不存在或已失效');

  const user = reviveUser(record);
  user.key = key;
  user.lastLoginAt = new Date().toISOString();
  await putUser(env, key, user);

  return { key, user };
}

/* ================= 文本清洗与计费 ================= */

// 去除所有空白字符（空格 / 换行 / 制表 / 全角空格 / 零宽字符），保留标点
function cleanText(raw) {
  return String(raw)
    .replace(/[\s\u3000\u00a0\u200b-\u200d\ufeff]+/g, '')
    .trim();
}

// 计费：按清洗后字数分阶梯，每 CHAR_UNIT 字 CHARGE_PER_UNIT 点，向上取整
function calcCost(charCount) {
  return Math.max(MIN_CHARGE, Math.ceil(charCount / CHAR_UNIT) * CHARGE_PER_UNIT);
}

/* ================= 评分规整 ================= */

function clampScore(value) {
  const num = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(num)) return null;
  return Math.min(100, Math.max(0, Math.round(num)));
}

// 白名单取值 + 0-100 整数钳制，缺项按中位 50 补齐，缺项过半视为无效输出
function sanitizeScores(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return null;
  const source = payload.scores && typeof payload.scores === 'object' ? payload.scores : payload;

  const scores = {};
  let matched = 0;
  for (const dimension of DIMENSION_KEYS) {
    const score = clampScore(source[dimension]);
    if (score === null) {
      scores[dimension] = 50;
    } else {
      scores[dimension] = score;
      matched += 1;
    }
  }
  return matched * 2 < DIMENSION_KEYS.length ? null : scores;
}

/* ================= 业务接口 ================= */

// POST /api/parse { key, text, name } -> { scores, characterName, balance, model, usage }
async function handleParse(request, env) {
  const body = await readJsonBody(request);
  const key = normalizeKey(body.key);

  if (typeof body.text !== 'string') {
    throw new HttpError(400, 'INVALID_TEXT', 'text 必须为字符串');
  }
  if (body.text.length > TEXT_MAX_LENGTH) {
    throw new HttpError(400, 'TEXT_TOO_LONG', `文本长度 ${body.text.length} 超过上限 ${TEXT_MAX_LENGTH} 字符`);
  }

  if (typeof body.name !== 'string' || !body.name.trim()) {
    throw new HttpError(400, 'INVALID_CHARACTER', '请先填写要侧写的角色名');
  }
  const characterName = body.name.trim().slice(0, 32);

  const mbtiRaw = typeof body.mbti === 'string' ? body.mbti.trim().toUpperCase() : '';
  const mbti = /^[EI][NS][TF][JP]$/.test(mbtiRaw) ? mbtiRaw : '';

  const text = cleanText(body.text);
  if (!text) throw new HttpError(400, 'EMPTY_TEXT', '文本清洗后为空');

  const user = await getUser(env, key);
  if (!user) throw new HttpError(404, 'USER_NOT_FOUND', '灵魂契约不存在或已失效');

  // 测试期：不限额、不扣费
  const cost = UNLIMITED_MODE ? 0 : calcCost(text.length);
  const balance = Number.isFinite(user.balance) ? user.balance : 0;

  if (!UNLIMITED_MODE && balance < cost) {
    throw new HttpError(402, 'INSUFFICIENT_BALANCE', `余额不足：本次消耗 ${cost} 点，当前余额 ${balance} 点`);
  }

  // 动态路由：<= 8000 字走 deepseek-chat，> 8000 字走 deepseek-reasoner
  const model = pickModel(text);

  const callOptions = {
    system: PARSE_SYSTEM_PROMPT,
    user: buildParseUserPrompt(text, characterName, mbti),
    model,
    json: true,
    temperature: PARSE_TEMPERATURE,
    maxTokens: PARSE_MAX_TOKENS,
  };

  let result;
  try {
    result = await callDeepSeek(env, callOptions);
  } catch (error) {
    // 空回复兜底：关掉 JSON 模式、换 chat 模型再试一次
    if (error && error.code === 'DEEPSEEK_EMPTY_CONTENT') {
      console.warn('[parse] 空回复，降级重试：', error.message);
      result = await callDeepSeek(env, {
        ...callOptions,
        model: MODEL_CHAT,
        json: false,
        temperature: 0.3,
        maxTokens: PARSE_MAX_TOKENS,
        user: `${callOptions.user}\n\n再次强调：只输出那个 JSON 对象本身，不要任何解释。`,
      });
    } else {
      throw error;
    }
  }

  const parsed = extractJson(result.content);

  // 「不是……而是……」「不是……，是……」一律重写一次
  const BANNED_REVIEW = /不是[^。！？\n]{0,30}(而是|，\s*是)/;
  let finalParsed = parsed;
  let finalResult = result;

  if (parsed && typeof parsed === 'object' && typeof parsed.review === 'string' && BANNED_REVIEW.test(parsed.review)) {
    const retry = await callDeepSeek(env, {
      ...callOptions,
      temperature: 0.35,
      user: `${callOptions.user}\n\n重要：批注里绝对不能出现「不是……而是……」「不是……，是……」这类句式，一旦想到就整段换一种说法重写。`,
    });
    const retryParsed = extractJson(retry.content);
    if (retryParsed && typeof retryParsed === 'object' && Array.isArray(Object.keys(retryParsed))) {
      finalParsed = retryParsed;
      finalResult = retry;
    }
  }

  if (finalParsed && typeof finalParsed === 'object' && !Array.isArray(finalParsed) && finalParsed.error === 'CHARACTER_NOT_FOUND') {
    throw new HttpError(
      400,
      'CHARACTER_NOT_FOUND',
      `文本里没找到「${characterName}」，请检查名字写法，或换一段有 ta 出场的文本`
    );
  }

  const scores = sanitizeScores(finalParsed);
  if (!scores) {
    throw new HttpError(
      502,
      'AI_OUTPUT_INVALID',
      `AI 返回的不是合法的 10 维 JSON（finish_reason=${result.finishReason || 'unknown'}），请重试`
    );
  }

  // 测试期不扣费，余额保持不变
  const remaining = UNLIMITED_MODE ? balance : Math.max(0, balance - cost);

  const profile = {
    code: await issueOcCode(env),
    name: characterName,
    mbti,
    scores,
    review: shortText(finalParsed.review, 520),
    dimensions: sanitizeDimensions(finalParsed.dimensions),
    createdAt: new Date().toISOString(),
    model: finalResult.model,
  };

  await putOc(env, profile);

  const records = Array.isArray(user.oc_records) ? [...user.oc_records] : [];
  if (!records.includes(profile.code)) records.push(profile.code);
  await putUser(env, key, { ...user, balance: remaining, oc_records: records });

  return {
    profile,
    model: finalResult.model,
    charCount: text.length,
    cost,
    balance: remaining,
    usage: result.usage,
    latencyMs: result.latencyMs,
  };
}

// POST /api/my-ocs { key } -> { ocs: [profile] }
async function handleMyOcs(request, env) {
  const body = await readJsonBody(request);
  const key = normalizeKey(body.key);

  const record = await getUser(env, key);
  if (!record) throw new HttpError(404, 'USER_NOT_FOUND', '灵魂契约不存在或已失效');
  const user = reviveUser(record);

  const ocs = [];
  for (const code of user.oc_records) {
    const oc = await getOc(env, code);
    if (oc) ocs.push(oc);
  }
  ocs.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));

  return { ocs };
}

// POST /api/oc-import { key, code } -> { oc }
async function handleImportOc(request, env) {
  const body = await readJsonBody(request);
  const key = normalizeKey(body.key);

  const raw = typeof body.code === 'string' ? body.code.trim().toUpperCase() : '';
  const code = raw.startsWith(`${OC_PREFIX}-`) ? raw : `${OC_PREFIX}-${raw}`;
  if (!OC_CODE_PATTERN.test(code)) {
    throw new HttpError(400, 'INVALID_OC_CODE', '角色码格式应为 OC-XXXX');
  }

  const record = await getUser(env, key);
  if (!record) throw new HttpError(404, 'USER_NOT_FOUND', '灵魂契约不存在或已失效');
  const user = reviveUser(record);

  const oc = await getOc(env, code);
  if (!oc) throw new HttpError(404, 'OC_NOT_FOUND', '这个角色码不存在或已失效');

  if (!user.oc_records.includes(code)) {
    await putUser(env, key, { ...user, oc_records: [...user.oc_records, code] });
  }

  return { oc };
}

/* ================= 反馈与券码 ================= */

// 只统计字母与数字，纯表情 / 纯符号会得到极低值
function countMeaningful(text) {
  const matched = text.match(/[\p{L}\p{N}]/gu);
  return matched ? matched.length : 0;
}

function randomCoupon() {
  const bytes = new Uint8Array(4);
  crypto.getRandomValues(bytes);
  const chars = Array.from(bytes, (byte) => KEY_ALPHABET[byte % KEY_ALPHABET.length]);
  return `${COUPON_PREFIX}-${chars.join('')}`;
}

function issueCoupon(user) {
  const owned = new Set(Array.isArray(user.coupons) ? user.coupons : []);
  for (let attempt = 0; attempt < COUPON_ISSUE_ATTEMPTS; attempt += 1) {
    const code = randomCoupon();
    if (!owned.has(code)) return code;
  }
  throw new HttpError(500, 'COUPON_ISSUE_FAILED', '券码生成失败，请重试');
}

function normalizeCoupon(raw) {
  if (typeof raw !== 'string' || !raw.trim()) {
    throw new HttpError(400, 'INVALID_COUPON', '请输入券码');
  }
  const compact = raw.trim().toUpperCase().replace(/[\s_]+/g, '-');
  const normalized = compact.startsWith(`${COUPON_PREFIX}-`) ? compact : `${COUPON_PREFIX}-${compact}`;
  if (!COUPON_PATTERN.test(normalized)) {
    throw new HttpError(400, 'INVALID_COUPON_FORMAT', '券码格式应为 VIP-XXXX');
  }
  return normalized;
}

// POST /api/feedback { key, text } -> { coupon, coupons, balance }
async function handleFeedback(request, env, info) {
  const body = await readJsonBody(request);
  const key = normalizeKey(body.key);

  if (typeof body.text !== 'string') {
    throw new HttpError(400, 'INVALID_FEEDBACK', '反馈内容必须为字符串');
  }

  const text = body.text.trim();
  const length = [...text].length;
  if (length < FEEDBACK_MIN_LENGTH || length > FEEDBACK_MAX_LENGTH) {
    throw new HttpError(
      400,
      'INVALID_FEEDBACK_LENGTH',
      `反馈需为 ${FEEDBACK_MIN_LENGTH}-${FEEDBACK_MAX_LENGTH} 字，当前 ${length} 字`
    );
  }
  if (countMeaningful(text) < FEEDBACK_MIN_MEANINGFUL) {
    throw new HttpError(400, 'FEEDBACK_TOO_SYMBOLIC', '请填写真实的文字反馈，纯表情或纯符号不予通过');
  }

  const record = await getUser(env, key);
  if (!record) throw new HttpError(404, 'USER_NOT_FOUND', '灵魂契约不存在或已失效');
  const user = reviveUser(record);

  const coupon = issueCoupon(user);
  const coupons = [...user.coupons, coupon];
  await putUser(env, key, { ...user, coupons });

  // Server酱通知：用 waitUntil 异步发送，不阻塞响应，也不影响发券结果
  const pushTask = pushToServerChan(env, {
    title: `新反馈 · ${user.nickname}`,
    content: [
      `**昵称**：${user.nickname}`,
      `**密钥**：${key}`,
      `**发放券码**：${coupon}`,
      `**字数**：${length}`,
      '',
      '---',
      '',
      text,
    ].join('\n\n'),
  }).catch(() => null);

  const workerCtx = info && info.ctx;
  if (workerCtx && typeof workerCtx.waitUntil === 'function') {
    workerCtx.waitUntil(pushTask);
  } else {
    await pushTask;
  }

  return { coupon, coupons, balance: user.balance, reward: COUPON_REWARD };
}

// POST /api/oc-delete { key, code } -> { deleted }
async function handleDeleteOc(request, env) {
  const body = await readJsonBody(request);
  const key = normalizeKey(body.key);

  const raw = typeof body.code === 'string' ? body.code.trim().toUpperCase() : '';
  const code = raw.startsWith(`${OC_PREFIX}-`) ? raw : `${OC_PREFIX}-${raw}`;
  if (!OC_CODE_PATTERN.test(code)) {
    throw new HttpError(400, 'INVALID_OC_CODE', '角色码格式应为 OC-XXXX');
  }

  const record = await getUser(env, key);
  if (!record) throw new HttpError(404, 'USER_NOT_FOUND', '灵魂契约不存在或已失效');
  const user = reviveUser(record);

  const records = user.oc_records.filter((item) => item !== code);
  await putUser(env, key, { ...user, oc_records: records });

  return { deleted: code };
}

// POST /api/use-coupon { key, coupon } -> { coupon, coupons, balance, reward }
async function handleUseCoupon(request, env) {
  const body = await readJsonBody(request);
  const key = normalizeKey(body.key);
  const coupon = normalizeCoupon(body.coupon);

  const record = await getUser(env, key);
  if (!record) throw new HttpError(404, 'USER_NOT_FOUND', '灵魂契约不存在或已失效');
  const user = reviveUser(record);

  const index = user.coupons.indexOf(coupon);
  if (index === -1) throw new HttpError(400, 'COUPON_NOT_FOUND', '券码无效或已被使用');

  const coupons = user.coupons.slice();
  coupons.splice(index, 1);

  const balance = (Number.isFinite(user.balance) ? user.balance : 0) + COUPON_REWARD;
  await putUser(env, key, { ...user, coupons, balance });

  return { coupon, coupons, balance, reward: COUPON_REWARD };
}

/* ================= 路由 ================= */

function compilePath(pattern) {
  const keys = [];
  const source = pattern
    .split('/')
    .map((segment) => {
      if (segment.startsWith(':')) {
        keys.push(segment.slice(1));
        return '([^/]+)';
      }
      return segment.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    })
    .join('/');
  return { regex: new RegExp(`^${source}$`), keys };
}

const ROUTES = [
  { method: 'POST', path: '/api/auth', handler: handleAuth },
  { method: 'POST', path: '/api/login', handler: handleLogin },
  { method: 'POST', path: '/api/parse', handler: handleParse },
  { method: 'POST', path: '/api/my-ocs', handler: handleMyOcs },
  { method: 'POST', path: '/api/oc-import', handler: handleImportOc },
  { method: 'POST', path: '/api/oc-delete', handler: handleDeleteOc },
  { method: 'POST', path: '/api/feedback', handler: handleFeedback },
  { method: 'POST', path: '/api/use-coupon', handler: handleUseCoupon },
].map((route) => ({ ...route, ...compilePath(route.path) }));

function resolveRoute(method, pathname) {
  const path = pathname.length > 1 ? pathname.replace(/\/+$/, '') || '/' : pathname;
  const allowed = new Set();

  for (const route of ROUTES) {
    const matched = route.regex.exec(path);
    if (!matched) continue;

    allowed.add(route.method);
    if (route.method !== method) continue;

    const params = {};
    route.keys.forEach((name, index) => {
      let value = matched[index + 1] || '';
      try {
        value = decodeURIComponent(value);
      } catch {
        /* 解码失败时保留原文 */
      }
      params[name] = value;
    });
    return { route, params };
  }

  if (allowed.size > 0) {
    throw new HttpError(405, 'METHOD_NOT_ALLOWED', `${method} 不被该接口支持`, {
      Allow: [...allowed, 'OPTIONS'].join(', '),
    });
  }
  throw new HttpError(404, 'NOT_FOUND', `接口不存在：${path}`);
}

/* ================= 入口 ================= */

export default {
  async fetch(request, env, ctx) {
    const startedAt = Date.now();
    const url = new URL(request.url);

    // 预检请求
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: new Headers(CORS_HEADERS) });
    }

    let response;
    try {
      const { route, params } = resolveRoute(request.method, url.pathname);
      const data = await route.handler(request, env, { params, ctx, url });
      response = ok(data);
    } catch (error) {
      const known = typeof error?.code === 'string';
      if (!known || (Number.isInteger(error.status) && error.status >= 500)) {
        console.error('[worker]', error?.stack || error);
      }
      response = fail(error);
    }

    console.log(
      JSON.stringify({
        time: new Date().toISOString(),
        method: request.method,
        path: url.pathname,
        status: response.status,
        durationMs: Date.now() - startedAt,
        ray: request.headers.get('cf-ray') || null,
      })
    );

    return response;
  },
};

