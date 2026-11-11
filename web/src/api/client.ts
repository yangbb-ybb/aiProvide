/**
 * 后端 API 客户端。
 *
 * SSE 用 fetch + ReadableStream 手写解析：EventSource 只支持 GET，
 * 而 /api/chat 需要 POST body。
 */

export interface Citation {
  index: number
  source_path: string
  heading_path: string[]
  page: number | null
  score: number
  preview: string
  title?: string
  chunk_id?: string
  scores?: Record<string, number>
}

/** 检索调试面板用的召回块。 */
export interface DebugChunk {
  chunk_id: string
  doc_id: string
  content: string
  source_path: string
  source_type: string
  title: string
  heading_path: string[]
  page: number | null
  chunk_idx: number
  score: number
  scores: Record<string, number>
}

/** /api/search 的完整响应。 */
export interface SearchResponse {
  query: string
  top_n: number
  count: number
  results: DebugChunk[]
}

/** /api/documents 列表项。 */
export interface DocumentSummary {
  doc_id: string
  source_path: string
  source_type: string
  title: string
  chunks: number
  mtime: number
}

export interface ChatHistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

export type ChatEvent =
  | { type: 'delta'; text: string }
  | { type: 'citation'; citations: Citation[] }
  | { type: 'done'; session_id?: string }
  | { type: 'error'; message: string; session_id?: string }

export interface Health {
  status_code: number
  ollama: string
  chat_model: string
  models: string[]
  embed_model: string
  vector_dim: number
  rerank_model: string
  rerank_enabled: boolean
}

const BASE = '/api'

export async function fetchHealth(): Promise<Health> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error(`健康检查失败：HTTP ${res.status}`)
  return (await res.json()) as Health
}

/** 文档列表。 */
export async function fetchDocuments(): Promise<DocumentSummary[]> {
  const res = await fetch(`${BASE}/documents`)
  if (!res.ok) throw new Error(`文档列表获取失败：HTTP ${res.status}`)
  return (await res.json()) as DocumentSummary[]
}

/** 上传文档并入库。 */
export async function uploadDocument(file: File): Promise<void> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/documents`, { method: 'POST', body: form })
  if (!res.ok) {
    // 后端返回 detail 字段时把中文提示带出来，比 HTTP 500 有用
    const body = await res.json().catch(() => null)
    const detail = body && typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`
    throw new Error(detail)
  }
}

/** 通过 URL 导入网页并入库。 */
export async function ingestUrl(url: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const detail = body && typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`
    throw new Error(detail)
  }
}

/** 删除文档。 */
export async function deleteDocument(docId: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const detail = body && typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`
    throw new Error(detail)
  }
}

/** 检索调试：只看召回结果，不经过模型。 */
export async function searchChunks(query: string, topK = 10): Promise<SearchResponse> {
  const res = await fetch(`${BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const detail = body && typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`
    throw new Error(detail)
  }
  return (await res.json()) as SearchResponse
}

/** 发起流式问答，逐个吐出后端事件。 */
export async function* chatStream(
  question: string,
  history: ChatHistoryMessage[] = [],
  sessionId?: string,
  topK?: number,
): AsyncGenerator<ChatEvent> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (sessionId) headers['X-Session-ID'] = sessionId
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ question, history, session_id: sessionId ?? null, top_k: topK ?? null }),
  })
  if (!res.ok) {
    throw new Error(`请求失败：HTTP ${res.status}`)
  }
  if (!res.body) {
    throw new Error('当前浏览器不支持流式响应')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE 以空行分隔事件帧
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      const line = frame.trim()
      if (!line.startsWith('data:')) continue
      const payload = line.slice(5).trim()
      if (!payload) continue
      try {
        yield JSON.parse(payload) as ChatEvent
      } catch {
        // 跳过解析不了的帧，不要因为一帧坏数据打断整段回答
      }
    }
  }
}
