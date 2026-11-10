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
}

export type ChatEvent =
  | { type: 'delta'; text: string }
  | { type: 'citation'; citations: Citation[] }
  | { type: 'done' }
  | { type: 'error'; message: string }

export interface Health {
  status: string
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

/** 发起流式问答，逐个吐出后端事件。 */
export async function* chatStream(question: string, topK?: number): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK ?? null }),
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
