<script setup lang="ts">
import { ref } from 'vue'

interface DebugChunk {
  chunk_id: string
  source_path: string
  heading_path: string[]
  page: number | null
  content: string
  scores: Record<string, number>
}

const query = ref('')
const results = ref<DebugChunk[]>([])
const error = ref('')
const loading = ref(false)

async function run() {
  if (!query.value.trim()) return
  loading.value = true
  error.value = ''
  results.value = []
  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query.value, top_k: 10 }),
    })
    if (res.status === 501) {
      error.value = '检索调试接口在 M1 阶段实现 —— 先把检索链路跑通再回来调参。'
      return
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    results.value = (await res.json()) as DebugChunk[]
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="debug">
    <div class="head">
      <h2>检索调试</h2>
      <p class="hint">
        看召回块、各路分数（dense / bm25 / rrf / rerank）—— 调分块大小和 top-k 时靠这个判断，
        而不是靠感觉。
      </p>
    </div>

    <div class="bar">
      <input
        v-model="query"
        placeholder="输入查询，查看召回结果与分数"
        :disabled="loading"
        @keydown.enter="run"
      />
      <button class="primary" :disabled="loading || !query.trim()" @click="run">
        {{ loading ? '检索中…' : '检索' }}
      </button>
    </div>

    <p v-if="error" class="error">{{ error }}</p>

    <div v-if="results.length" class="results">
      <div v-for="(r, i) in results" :key="r.chunk_id" class="item">
        <div class="item-head">
          <span class="rank">#{{ i + 1 }}</span>
          <span class="path">{{ r.source_path }}</span>
          <span v-if="r.heading_path.length" class="heading">{{ r.heading_path.join(' › ') }}</span>
          <span v-if="r.page !== null" class="page">第 {{ r.page }} 页</span>
          <span class="scores">
            <span v-for="(v, k) in r.scores" :key="k" class="score">
              {{ k }}={{ typeof v === 'number' ? v.toFixed(4) : v }}
            </span>
          </span>
        </div>
        <pre class="content">{{ r.content }}</pre>
      </div>
    </div>
  </div>
</template>

<style scoped>
.debug {
  flex: 1;
  min-width: 0;
  padding: 24px;
  overflow-y: auto;
}

h2 {
  margin: 0 0 6px;
  font-size: 16px;
}

.hint {
  margin: 0 0 18px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.8;
}

.bar {
  display: flex;
  gap: 8px;
  margin-bottom: 18px;
}

.bar input {
  flex: 1;
}

.primary {
  background: #1f6feb;
  border: 1px solid #1f6feb;
  border-radius: 6px;
  color: #fff;
  padding: 7px 18px;
  cursor: pointer;
}

.primary:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.error {
  color: #f85149;
  margin-bottom: 14px;
}

.item {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 10px;
  background: var(--panel);
}

.item-head {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 8px;
}

.rank {
  color: var(--accent);
  font-weight: 600;
}

.path {
  color: var(--text);
}

.scores {
  margin-left: auto;
  display: flex;
  gap: 10px;
}

.score {
  background: var(--hover);
  padding: 1px 7px;
  border-radius: 4px;
  font-variant-numeric: tabular-nums;
}

.content {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  line-height: 1.7;
  color: var(--text);
}
</style>
