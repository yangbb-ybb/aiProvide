import { defineStore } from 'pinia'
import { reactive, ref } from 'vue'

import { chatStream, type ChatHistoryMessage, type Citation } from '../api/client'

export interface Message {
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  error?: string
  progress?: string
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
    // 使用 reactive 代理承载正在生成的消息。这样每个 delta 写入后都会
    // 立即触发 Vue 更新，避免把普通对象当作“最后统一提交”的结果。
    const reply = reactive<Message>({ role: 'assistant', content: '', citations: [], progress: '正在准备回答…' })
    messages.value.push(reply)
    streaming.value = true
    // 引用虽然可能比正文更早从后端到达，但聊天界面应该先展示回答正文，
    // 等正文完成后再展示来源，避免用户看到一大块“引用来源”却还没有答案。
    let pendingCitations: Citation[] = []

    try {
      for await (const ev of chatStream(text, history, sessionId.value)) {
        if (ev.type === 'progress') {
          reply.progress = ev.message
        } else if (ev.type === 'delta') {
          reply.progress = undefined
          reply.content += ev.text
        } else if (ev.type === 'citation') {
          pendingCitations = ev.citations
        } else if (ev.type === 'done') {
          reply.citations = pendingCitations
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
