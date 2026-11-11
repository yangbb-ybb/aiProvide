<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'

import { useChatStore } from '../stores/chat'

const store = useChatStore()
const input = ref('')
const listEl = ref<HTMLElement | null>(null)
const shouldFollowOutput = ref(true)

function send() {
  const q = input.value
  if (!q.trim() || store.streaming) return
  input.value = ''
  void store.ask(q)
}

function onMessagesScroll() {
  const el = listEl.value
  if (!el) return
  // 只有用户本来就在底部附近时才跟随新内容；用户向上阅读时绝不抢滚动位置。
  shouldFollowOutput.value = el.scrollHeight - el.scrollTop - el.clientHeight < 48
}

// 流式输出时仅在用户未主动上翻时保持滚动到底部。
watch(
  () => store.messages.map((m) => `${m.content.length}:${m.progress ?? ''}:${m.citations.length}`).join(','),
  async () => {
    if (!shouldFollowOutput.value) return
    await nextTick()
    if (listEl.value) listEl.value.scrollTop = listEl.value.scrollHeight
  },
)
</script>

<template>
  <div class="chat">
    <div ref="listEl" class="messages" @scroll="onMessagesScroll">
      <div v-if="store.messages.length === 0" class="empty">
        <p>问点什么吧 —— 回答会基于你导入的资料，并标注来源。</p>
        <p class="hint">支持连续追问和不同说法；事实仍以知识库资料为准。</p>
      </div>

      <div v-for="(m, i) in store.messages" :key="i" :class="['msg', m.role]">
        <div class="bubble">
          <div v-if="m.role === 'assistant' && !m.content && m.progress" class="progress">{{ m.progress }}<span class="cursor">▍</span></div>
          <div v-if="m.content" class="content">{{ m.content }}<span v-if="m.role === 'assistant' && store.streaming && i === store.messages.length - 1" class="cursor">▍</span></div>

          <div v-if="m.error" class="error">{{ m.error }}</div>

          <div v-if="m.citations.length" class="citations">
            <div class="cite-title">引用来源</div>
            <div v-for="c in m.citations" :key="c.index" class="cite">
              <span class="idx">[{{ c.index }}]</span>
              <span class="path">{{ c.source_path }}</span>
              <span v-if="c.heading_path.length" class="heading">{{ c.heading_path.join(' › ') }}</span>
              <span v-if="c.page !== null" class="page">第 {{ c.page }} 页</span>
              <span class="score">{{ c.score.toFixed(3) }}</span>
              <div v-if="c.preview" class="preview">{{ c.preview }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="composer">
      <textarea
        v-model="input"
        rows="2"
        placeholder="输入问题，Ctrl/⌘ + Enter 发送"
        :disabled="store.streaming"
        @keydown.meta.enter.prevent="send"
        @keydown.ctrl.enter.prevent="send"
      ></textarea>
      <div class="actions">
        <button class="secondary" :disabled="store.messages.length === 0" @click="store.clear()">清空</button>
        <button class="primary" :disabled="store.streaming || !input.trim()" @click="send">
          {{ store.streaming ? '生成中…' : '发送' }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px 0;
}

.hint {
  font-size: 12px;
  opacity: 0.75;
}

.msg {
  display: flex;
  margin-bottom: 18px;
  padding: 0 20px;
}

.msg.user {
  justify-content: flex-end;
}

.bubble {
  max-width: 760px;
  width: 100%;
  padding: 12px 14px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--panel);
}

.msg.user .bubble {
  width: auto;
  max-width: 76%;
  background: #1f6feb26;
  border-color: #1f6feb55;
}

.progress {
  color: var(--muted);
  font-size: 13px;
}

.content {
  white-space: pre-wrap;
  word-break: break-word;
}

.cursor {
  opacity: 0.6;
  animation: blink 1s steps(2, start) infinite;
}

@keyframes blink {
  to {
    visibility: hidden;
  }
}

.error {
  margin-top: 8px;
  color: #f85149;
  font-size: 13px;
}

.citations {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px dashed var(--border);
  font-size: 12px;
  color: var(--muted);
}

.cite-title {
  margin-bottom: 6px;
  color: var(--text);
  font-weight: 600;
}

.cite {
  margin-bottom: 6px;
  line-height: 1.5;
}

.idx {
  color: var(--accent);
  font-weight: 600;
  margin-right: 4px;
}

.heading {
  margin-left: 8px;
  opacity: 0.85;
}

.page,
.score {
  margin-left: 8px;
  opacity: 0.6;
}

.preview {
  margin-top: 3px;
  padding-left: 18px;
  opacity: 0.75;
  white-space: pre-wrap;
}

.composer {
  border-top: 1px solid var(--border);
  padding: 14px 20px;
  background: var(--panel);
}

textarea {
  width: 100%;
  resize: vertical;
  line-height: 1.6;
}

.actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 10px;
}

.actions button {
  border-radius: 6px;
  padding: 7px 16px;
  cursor: pointer;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text);
}

.actions button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.actions .primary {
  background: #1f6feb;
  border-color: #1f6feb;
}

.actions .secondary:hover:not(:disabled) {
  background: var(--hover);
}
</style>
