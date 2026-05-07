import { contextBridge, ipcRenderer } from 'electron'

export interface ElectronBridge {
  dialog: {
    openFile: (options?: any) => Promise<string | null>
    openDirectory: () => Promise<string | null>
    saveFile: (options?: any) => Promise<string | null>
  }
  python: {
    run: (args: string[]) => Promise<any>
    runStream: (args: string[]) => Promise<any>
    cancel: () => Promise<boolean>
    onStream: (callback: (data: any) => void) => () => void
  }
  fs: {
    readFile: (path: string) => Promise<{ success: boolean; content?: string; error?: string }>
    writeFile: (path: string, content: string) => Promise<{ success: boolean; error?: string }>
    exists: (path: string) => Promise<boolean>
  }
  path: {
    home: () => Promise<string>
    join: (...parts: string[]) => Promise<string>
  }
  app: {
    info: () => Promise<{ platform: string; arch: string; isDev: boolean; version: string }>
  }
}

const bridge: ElectronBridge = {
  dialog: {
    openFile: (options) => ipcRenderer.invoke('dialog:openFile', options),
    openDirectory: () => ipcRenderer.invoke('dialog:openDirectory'),
    saveFile: (options) => ipcRenderer.invoke('dialog:saveFile', options),
  },
  python: {
    run: (args) => ipcRenderer.invoke('python:run', args),
    runStream: (args) => ipcRenderer.invoke('python:runStream', args),
    cancel: () => ipcRenderer.invoke('python:cancel'),
    onStream: (callback) => {
      const handler = (_event: any, data: any) => callback(data)
      ipcRenderer.on('python:stream', handler)
      return () => ipcRenderer.removeListener('python:stream', handler)
    },
  },
  fs: {
    readFile: (path) => ipcRenderer.invoke('fs:readFile', path),
    writeFile: (path, content) => ipcRenderer.invoke('fs:writeFile', path, content),
    exists: (path) => ipcRenderer.invoke('fs:exists', path),
  },
  path: {
    home: () => ipcRenderer.invoke('path:home'),
    join: (...parts) => ipcRenderer.invoke('path:join', ...parts),
  },
  app: {
    info: () => ipcRenderer.invoke('app:info'),
  },
}

contextBridge.exposeInMainWorld('electron', bridge)
