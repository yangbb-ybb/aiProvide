<script setup lang="ts">
import { onMounted, ref } from 'vue'

import {
  deleteDocument,
  fetchDocuments,
  ingestUrl,
  uploadDocument,
  type DocumentSummary,
} from '../api/client'

const docs = ref<DocumentSummary[]>([])
const loading = ref(false)
const error = ref('')
const notice = ref('')
const uploading = ref(false)
const url = ref('')
const ingestingUrl = ref(false)
const deletingId = ref('')

async function load() {
  loading.value = true
  error.value = ''
  try {
    docs.value = await fetchDocuments()
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
  notice.value = ''
  try {
    await uploadDocument(file)
    notice.value = `已导入 ${file.name}`
    await load()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    uploading.value = false
    input.value = ''
  }
}

async function onIngestUrl() {
  const value = url.value.trim()
  if (!value || ingestingUrl.value) return
  ingestingUrl.value = true
  error.value = ''
  notice.value = ''
  try {
    await ingestUrl(value)
    notice.value = `已导入网页 ${value}`
    url.value = ''
    await load()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    ingestingUrl.value = false
  }
}

async function onDelete(doc: DocumentSummary) {
  if (!confirm(`确定删除「${doc.title || doc.source_path}」及其 ${doc.chunks} 个块？`)) return
  deletingId.value = doc.doc_id
  error.value = ''
  try {
    await deleteDocument(doc.doc_id)
    notice.value = `已删除 ${doc.title || doc.source_path}`
    await load()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    deletingId.value = ''
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

    <form class="url-form" @submit.prevent="onIngestUrl">
      <input
        v-model="url"
        type="url"
        placeholder="粘贴网页地址，例如 https://example.com/article"
        :disabled="ingestingUrl"
        required
      />
      <button class="primary" type="submit" :disabled="ingestingUrl || !url.trim()">
        {{ ingestingUrl ? '导入中…' : '导入网址' }}
      </button>
    </form>

    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="notice" class="notice">{{ notice }}</p>

    <div v-if="loading" class="empty">加载中…</div>
    <div v-else-if="docs.length === 0" class="empty">
      <p>还没有文档。</p>
      <p class="hint">
        在这里上传 md / pdf / docx / xlsx / pptx，或直接贴一个网址。<br />
        也可以把文件丢进 <code>server/data/uploads/</code>，然后用
        <code>uv run python -m app.eval.run</code> 之外的入库脚本手动处理。
      </p>
    </div>

    <table v-else class="table">
      <thead>
        <tr>
          <th>文件</th>
          <th>类型</th>
          <th>块数</th>
          <th>修改时间</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="d in docs" :key="d.doc_id">
          <td :title="d.source_path">{{ d.title || d.source_path }}</td>
          <td><span class="type">{{ d.source_type }}</span></td>
          <td>{{ d.chunks }}</td>
          <td>{{ new Date(d.mtime * 1000).toLocaleString('zh-CN') }}</td>
          <td>
            <button
              class="del"
              :disabled="deletingId === d.doc_id"
              @click="onDelete(d)"
            >
              {{ deletingId === d.doc_id ? '删除中…' : '删除' }}
            </button>
          </td>
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

.url-form {
  display: flex;
  gap: 8px;
  margin-bottom: 18px;
}

.url-form input {
  flex: 1;
}

.primary {
  background: #1f6feb;
  border: 1px solid #1f6feb;
  border-radius: 6px;
  color: #fff;
  padding: 6px 14px;
  cursor: pointer;
  font-size: 13px;
}

.primary:disabled {
  opacity: 0.45;
  cursor: not-allowed;
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

.notice {
  color: #3fb950;
  margin-bottom: 12px;
}

.del {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 5px;
  color: var(--muted);
  padding: 3px 10px;
  cursor: pointer;
  font-size: 12px;
}

.del:hover:not(:disabled) {
  color: #f85149;
  border-color: #f85149;
}

.del:disabled {
  opacity: 0.45;
  cursor: not-allowed;
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
