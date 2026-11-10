<script setup lang="ts">
import { onMounted, ref } from 'vue'

import { fetchHealth, type Health } from './api/client'
import ChatView from './views/ChatView.vue'
import LibraryView from './views/LibraryView.vue'
import RetrievalDebugView from './views/RetrievalDebugView.vue'

const tabs = [
  { key: 'chat', label: '问答' },
  { key: 'library', label: '知识库' },
  { key: 'debug', label: '检索调试' },
] as const

type TabKey = (typeof tabs)[number]['key']

const active = ref<TabKey>('chat')
const health = ref<Health | null>(null)
const healthError = ref('')

onMounted(async () => {
  try {
    health.value = await fetchHealth()
  } catch (e) {
    healthError.value = e instanceof Error ? e.message : String(e)
  }
})
</script>

<template>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <span class="logo">◈</span>
        <span>本地 RAG 知识库</span>
      </div>

      <nav class="tabs">
        <button
          v-for="t in tabs"
          :key="t.key"
          :class="['tab', { active: active === t.key }]"
          @click="active = t.key"
        >
          {{ t.label }}
        </button>
      </nav>

      <div class="status">
        <template v-if="health">
          <span class="dot" :class="health.ollama === 'ok' ? 'ok' : 'bad'"></span>
          <span class="status-text">{{ health.ollama === 'ok' ? health.chat_model : '模型未就绪' }}</span>
        </template>
        <template v-else-if="healthError">
          <span class="dot bad"></span>
          <span class="status-text" :title="healthError">后端未连接</span>
        </template>
        <span v-else class="status-text">检测中…</span>
      </div>
    </header>

    <main class="content">
      <ChatView v-if="active === 'chat'" />
      <LibraryView v-else-if="active === 'library'" />
      <RetrievalDebugView v-else />
    </main>
  </div>
</template>

<style scoped>
.topbar {
  display: flex;
  align-items: center;
  gap: 24px;
  padding: 0 20px;
  height: 52px;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
}

.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: 14px;
  white-space: nowrap;
}

.logo {
  color: var(--accent);
}

.tabs {
  display: flex;
  gap: 4px;
  flex: 1;
}

.tab {
  background: transparent;
  border: none;
  color: var(--muted);
  padding: 6px 14px;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
}

.tab:hover {
  color: var(--text);
  background: var(--hover);
}

.tab.active {
  color: var(--text);
  background: var(--hover);
}

.status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--muted);
  white-space: nowrap;
}

.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--muted);
}

.dot.ok {
  background: #3fb950;
}

.dot.bad {
  background: #f85149;
}

.content {
  flex: 1;
  min-height: 0;
  display: flex;
}
</style>
