import { defineStore } from 'pinia'
import { ref } from 'vue'

import { chatStream, type ChatHistoryMessage, type Citation } from '../api/client'

export interface Message {
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  error?: string
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const streaming = ref(false)
  const sessionId = ref<string | undefined>(undefined)

  async function ask(question: string) {
    const text = question.trim()
    if (!text || streaming.value) return

    // 先截取已有对话，再追加本轮问题，避免把当前问题重复放进 history。
    // 只发送最近 6 条有效消息，控制本地小模型的上下文和查询改写耗时。
    const history: ChatHistoryMessage[] = messages.value
      .filter((message) => message.content.trim() && !message.error)
      .slice(-6)
      .map((message) => ({ role: message.role, content: message.content }))

    messages.value.push({ role: 'user', content: text, citations: [] })
    const reply: Message = { role: 'assistant', content: '', citations: [] }
    messages.value.push(reply)
    streaming.value = true

    try {
      for await (const ev of chatStream(text, history, sessionId.value)) {
        if (ev.type === 'delta') {
          reply.content += ev.text
        } else if (ev.type === 'citation') {
          reply.citations = ev.citations
        } else if (ev.type === 'done') {
          if (ev.session_id) sessionId.value = ev.session_id
        } else if (ev.type === 'error') {
          if (ev.session_id) sessionId.value = ev.session_id
          reply.error = ev.message
        }
      }
    } catch (e) {
      reply.error = e instanceof Error ? e.message : String(e)
    } finally {
      streaming.value = false
    }
  }

  function clear() {
    messages.value = []
    sessionId.value = undefined
  }

  return { messages, streaming, sessionId, ask, clear }
})
