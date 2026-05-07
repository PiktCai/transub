import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import * as path from 'path'
import * as fs from 'fs'

let mainWindow: BrowserWindow | null = null
let activeProcess: ChildProcess | null = null

const isDev = !app.isPackaged
const repoRoot = isDev ? path.resolve(process.cwd(), '..') : process.cwd()

app.disableHardwareAcceleration()

function findTransubCommand(): string {
  return 'uv run transub'
}

function commandParts(command: string): [string, string[]] {
  const [bin, ...args] = command.split(' ')
  return [bin, args]
}

function processEnv() {
  const pathValue = process.env.PATH || ''
  const extraPaths = ['/opt/homebrew/bin', '/usr/local/bin', '/usr/bin', '/bin']
  return {
    ...process.env,
    PATH: [...extraPaths, pathValue].filter(Boolean).join(path.delimiter),
    PYTHONUNBUFFERED: '1',
    HF_HUB_ENABLE_HF_TRANSFER: process.env.HF_HUB_ENABLE_HF_TRANSFER || '1',
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

app.whenReady().then(createWindow)

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow()
  }
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

ipcMain.handle('python:run', async (_event, args: string[]) => {
  const [bin, baseArgs] = commandParts(findTransubCommand())
  const allArgs = [...baseArgs, ...args]

  return new Promise((resolve, reject) => {
    const proc = spawn(bin, allArgs, {
      shell: false,
      cwd: repoRoot,
      env: processEnv(),
    })

    let stdout = ''
    let stderr = ''

    proc.stdout.on('data', (data: Buffer) => {
      const text = data.toString()
      stdout += text
      mainWindow?.webContents.send('python:stdout', text)
    })

    proc.stderr.on('data', (data: Buffer) => {
      const text = data.toString()
      stderr += text
      mainWindow?.webContents.send('python:stderr', text)
    })

    proc.on('close', (code) => {
      if (code === 0) {
        resolve({ success: true, stdout, stderr, code })
      } else {
        resolve({ success: false, stdout, stderr, code })
      }
    })

    proc.on('error', (err) => {
      reject({ success: false, error: err.message })
    })
  })
})

ipcMain.handle('python:runStream', async (_event, args: string[]) => {
  const [bin, baseArgs] = commandParts(findTransubCommand())
  const allArgs = [...baseArgs, ...args]

  const proc = spawn(bin, allArgs, {
    shell: false,
    cwd: repoRoot,
    env: processEnv(),
  })

  activeProcess = proc

  return new Promise((resolve) => {
    let stdout = ''
    let stderr = ''

    proc.stdout.on('data', (data: Buffer) => {
      const text = data.toString()
      stdout += text
      mainWindow?.webContents.send('python:stream', { type: 'stdout', data: text })
    })

    proc.stderr.on('data', (data: Buffer) => {
      const text = data.toString()
      stderr += text
      mainWindow?.webContents.send('python:stream', { type: 'stderr', data: text })
    })

    proc.on('close', (code) => {
      activeProcess = null
      mainWindow?.webContents.send('python:stream', { type: 'done', code })
      resolve({ success: code === 0, stdout, stderr, code })
    })

    proc.on('error', (err) => {
      activeProcess = null
      mainWindow?.webContents.send('python:stream', { type: 'error', error: err.message })
      resolve({ success: false, error: err.message })
    })
  })
})

ipcMain.handle('python:cancel', async () => {
  if (activeProcess) {
    activeProcess.kill('SIGTERM')
    activeProcess = null
    return true
  }
  return false
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

ipcMain.handle('fs:exists', async (_event, filePath: string) => {
  return fs.existsSync(filePath)
})

ipcMain.handle('path:home', () => {
  return process.env.HOME || process.env.USERPROFILE || ''
})

ipcMain.handle('path:join', (_event, ...parts: string[]) => {
  return path.join(...parts)
})

ipcMain.handle('app:info', () => {
  return {
    platform: process.platform,
    arch: process.arch,
    isDev,
    version: app.getVersion(),
  }
})
