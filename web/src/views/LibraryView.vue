<script setup lang="ts">
import { onMounted, ref } from 'vue'

interface DocumentSummary {
  doc_id: string
  source_path: string
  source_type: string
  title: string
  chunks: number
  mtime: number
}

const docs = ref<DocumentSummary[]>([])
const loading = ref(false)
const error = ref('')
const uploading = ref(false)

async function load() {
  loading.value = true
  error.value = ''
  try {
    const res = await fetch('/api/documents')
    if (res.status === 501) {
      docs.value = []
      return
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    docs.value = (await res.json()) as DocumentSummary[]
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

async function onUpload(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  uploading.value = true
  error.value = ''
  try {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch('/api/documents', { method: 'POST', body: form })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    await load()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    uploading.value = false
    input.value = ''
  }
}

onMounted(load)
</script>

<template>
  <div class="library">
    <div class="head">
      <h2>知识库</h2>
      <div class="tools">
        <label class="upload">
          <input type="file" hidden @change="onUpload" />
          <span>{{ uploading ? '上传中…' : '上传文档' }}</span>
        </label>
        <button class="ghost" @click="load">刷新</button>
      </div>
    </div>

    <p v-if="error" class="error">{{ error }}</p>

    <div v-if="loading" class="empty">加载中…</div>
    <div v-else-if="docs.length === 0" class="empty">
      <p>还没有文档。</p>
      <p class="hint">
        M1 实现后可以在这里上传 md / pdf / docx / xlsx / pptx，或直接贴一个网址。<br />
        也可以把文件丢进 <code>server/data/uploads/</code>。
      </p>
    </div>

    <table v-else class="table">
      <thead>
        <tr>
          <th>文件</th>
          <th>类型</th>
          <th>块数</th>
          <th>修改时间</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="d in docs" :key="d.doc_id">
          <td :title="d.source_path">{{ d.title || d.source_path }}</td>
          <td><span class="type">{{ d.source_type }}</span></td>
          <td>{{ d.chunks }}</td>
          <td>{{ new Date(d.mtime * 1000).toLocaleString('zh-CN') }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.library {
  flex: 1;
  min-width: 0;
  padding: 24px;
  overflow-y: auto;
}

.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 18px;
}

h2 {
  margin: 0;
  font-size: 16px;
}

.tools {
  display: flex;
  gap: 8px;
}

.upload {
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 14px;
  cursor: pointer;
  font-size: 13px;
}

.upload:hover {
  background: var(--hover);
}

.ghost {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  padding: 6px 14px;
  cursor: pointer;
  font-size: 13px;
}

.ghost:hover {
  background: var(--hover);
}

.hint {
  font-size: 12px;
  opacity: 0.75;
  line-height: 1.9;
}

code {
  background: var(--hover);
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 12px;
}

.error {
  color: #f85149;
  margin-bottom: 12px;
}

.table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

th,
td {
  text-align: left;
  padding: 9px 12px;
  border-bottom: 1px solid var(--border);
}

th {
  color: var(--muted);
  font-weight: 500;
  font-size: 12px;
}

.type {
  background: var(--hover);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
}
</style>
