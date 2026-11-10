import { defineStore } from 'pinia'
import { ref } from 'vue'

import { chatStream, type Citation } from '../api/client'

export interface Message {
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  error?: string
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const streaming = ref(false)

  async function ask(question: string) {
    const text = question.trim()
    if (!text || streaming.value) return

    messages.value.push({ role: 'user', content: text, citations: [] })
    const reply: Message = { role: 'assistant', content: '', citations: [] }
    messages.value.push(reply)
    streaming.value = true

    try {
      for await (const ev of chatStream(text)) {
        if (ev.type === 'delta') {
          reply.content += ev.text
        } else if (ev.type === 'citation') {
          reply.citations = ev.citations
        } else if (ev.type === 'error') {
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
  }

  return { messages, streaming, ask, clear }
})
