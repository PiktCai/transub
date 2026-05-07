import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import * as path from 'path'
import * as fs from 'fs'
import * as http from 'http'

let mainWindow: BrowserWindow | null = null
let pythonProcess: ChildProcess | null = null
const API_PORT = 18789
const API_BASE = `http://127.0.0.1:${API_PORT}`

const isDev = !app.isPackaged
const repoRoot = isDev ? path.resolve(process.cwd(), '..') : process.cwd()

app.disableHardwareAcceleration()

function processEnv() {
  const pathValue = process.env.PATH || ''
  const extraPaths = ['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin', '/bin']
  return {
    ...process.env,
    PATH: [...extraPaths, pathValue].filter(Boolean).join(path.delimiter),
    PYTHONUNBUFFERED: '1',
  }
}

async function waitForServer(timeoutMs = 15000): Promise<boolean> {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`${API_BASE}/api/health`)
      if (res.ok) return true
    } catch {}
    await new Promise(r => setTimeout(r, 200))
  }
  return false
}

async function startPythonServer(): Promise<boolean> {
  const bin = 'uv'
  const args = ['run', '--extra', 'server', 'transub', 'serve', '--port', String(API_PORT)]

  pythonProcess = spawn(bin, args, {
    shell: false,
    cwd: repoRoot,
    env: processEnv(),
    stdio: ['ignore', 'pipe', 'pipe'],
  })

  pythonProcess.stdout?.on('data', (data: Buffer) => {
    console.log('[python]', data.toString().trim())
  })
  pythonProcess.stderr?.on('data', (data: Buffer) => {
    console.error('[python]', data.toString().trim())
  })
  pythonProcess.on('exit', (code) => {
    console.log('[python] exited with code', code)
    pythonProcess = null
  })

  return waitForServer()
}

function stopPythonServer() {
  if (pythonProcess) {
    pythonProcess.kill('SIGTERM')
    pythonProcess = null
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 960,
    minHeight: 640,
    title: 'Transub',
    backgroundColor: '#FFFCF0',
    webPreferences: {
      preload: path.join(__dirname, '../preload/preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  if (process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    mainWindow.loadFile(path.join(__dirname, '../../dist/index.html'))
  }

  mainWindow.webContents.on('render-process-gone', (_event, details) => {
    console.error('[renderer gone]', details)
  })

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

app.whenReady().then(async () => {
  const serverReady = await startPythonServer()
  if (!serverReady) {
    console.error('Failed to start Python server')
  }
  createWindow()
})

app.on('window-all-closed', () => {
  stopPythonServer()
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow()
  }
})

app.on('before-quit', () => {
  stopPythonServer()
})

async function apiCall(method: string, path: string, body?: any): Promise<any> {
  const opts: RequestInit = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body) opts.body = JSON.stringify(body)
  const res = await fetch(`${API_BASE}${path}`, opts)
  return res.json()
}

ipcMain.handle('api:health', async () => {
  return apiCall('GET', '/api/health')
})

ipcMain.handle('api:getConfig', async (_event, configPath?: string) => {
  const qs = configPath ? `?config_path=${encodeURIComponent(configPath)}` : ''
  return apiCall('GET', `/api/config${qs}`)
})

ipcMain.handle('api:updateConfig', async (_event, config: object, configPath?: string) => {
  return apiCall('PUT', '/api/config', { config, config_path: configPath })
})

ipcMain.handle('api:getAuth', async () => {
  return apiCall('GET', '/api/auth')
})

ipcMain.handle('api:saveAuth', async (_event, provider: string, data: { api_key?: string; api_base?: string }) => {
  return apiCall('PUT', `/api/auth/${provider}`, data)
})

ipcMain.handle('api:deleteAuth', async (_event, provider: string) => {
  return apiCall('DELETE', `/api/auth/${provider}`)
})

ipcMain.handle('api:prepareModel', async (_event, configPath?: string) => {
  return apiCall('POST', '/api/prepare-model', { config_path: configPath })
})

ipcMain.handle('api:runPipeline', async (_event, videoPath: string, options: { transcribeOnly?: boolean; configPath?: string; workDir?: string } = {}) => {
  return apiCall('POST', '/api/run', {
    video_path: videoPath,
    transcribe_only: options.transcribeOnly || false,
    config_path: options.configPath,
    work_dir: options.workDir,
  })
})

ipcMain.handle('api:cancelPipeline', async () => {
  return apiCall('POST', '/api/cancel')
})

ipcMain.handle('api:cacheStats', async () => {
  return apiCall('GET', '/api/cache/stats')
})

ipcMain.handle('api:clearCache', async () => {
  return apiCall('DELETE', '/api/cache')
})

ipcMain.handle('api:stream', async () => {
  return { url: `${API_BASE}/api/stream` }
})

ipcMain.handle('dialog:openFile', async (_event, options) => {
  if (!mainWindow) return null
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: options?.filters || [
      { name: 'Video Files', extensions: ['mp4', 'mkv', 'avi', 'mov', 'webm', 'flv', 'wmv'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  })
  return result.canceled ? null : result.filePaths[0]
})

ipcMain.handle('dialog:openDirectory', async () => {
  if (!mainWindow) return null
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
  })
  return result.canceled ? null : result.filePaths[0]
})

ipcMain.handle('dialog:saveFile', async (_event, options) => {
  if (!mainWindow) return null
  const result = await dialog.showSaveDialog(mainWindow, {
    filters: options?.filters || [
      { name: 'SRT Subtitles', extensions: ['srt'] },
      { name: 'VTT Subtitles', extensions: ['vtt'] },
    ],
  })
  return result.canceled ? null : result.filePath
})

ipcMain.handle('fs:readFile', async (_event, filePath: string) => {
  try {
    const content = fs.readFileSync(filePath, 'utf-8')
    return { success: true, content }
  } catch (err: any) {
    return { success: false, error: err.message }
  }
})

ipcMain.handle('fs:writeFile', async (_event, filePath: string, content: string) => {
  try {
    fs.mkdirSync(path.dirname(filePath), { recursive: true })
    fs.writeFileSync(filePath, content, 'utf-8')
    return { success: true }
  } catch (err: any) {
    return { success: false, error: err.message }
  }
})

ipcMain.handle('path:home', () => {
  return process.env.HOME || process.env.USERPROFILE || ''
})
