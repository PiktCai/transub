import { contextBridge, ipcRenderer } from 'electron'

export interface ElectronBridge {
  api: {
    health: () => Promise<any>
    getConfig: (configPath?: string) => Promise<any>
    updateConfig: (config: object, configPath?: string) => Promise<any>
    getAuth: () => Promise<any>
    saveAuth: (provider: string, data: { api_key?: string; api_base?: string }) => Promise<any>
    deleteAuth: (provider: string) => Promise<any>
    prepareModel: (configPath?: string) => Promise<any>
    runPipeline: (videoPath: string, options?: any) => Promise<any>
    cancelPipeline: () => Promise<any>
    cacheStats: () => Promise<any>
    clearCache: () => Promise<any>
    streamUrl: () => Promise<{ url: string }>
  }
  dialog: {
    openFile: (options?: any) => Promise<string | null>
    openDirectory: () => Promise<string | null>
    saveFile: (options?: any) => Promise<string | null>
  }
  fs: {
    readFile: (path: string) => Promise<{ success: boolean; content?: string; error?: string }>
    writeFile: (path: string, content: string) => Promise<{ success: boolean; error?: string }>
  }
  path: {
    home: () => Promise<string>
  }
}

const bridge: ElectronBridge = {
  api: {
    health: () => ipcRenderer.invoke('api:health'),
    getConfig: (configPath) => ipcRenderer.invoke('api:getConfig', configPath),
    updateConfig: (config, configPath) => ipcRenderer.invoke('api:updateConfig', config, configPath),
    getAuth: () => ipcRenderer.invoke('api:getAuth'),
    saveAuth: (provider, data) => ipcRenderer.invoke('api:saveAuth', provider, data),
    deleteAuth: (provider) => ipcRenderer.invoke('api:deleteAuth', provider),
    prepareModel: (configPath) => ipcRenderer.invoke('api:prepareModel', configPath),
    runPipeline: (videoPath, options) => ipcRenderer.invoke('api:runPipeline', videoPath, options),
    cancelPipeline: () => ipcRenderer.invoke('api:cancelPipeline'),
    cacheStats: () => ipcRenderer.invoke('api:cacheStats'),
    clearCache: () => ipcRenderer.invoke('api:clearCache'),
    streamUrl: () => ipcRenderer.invoke('api:stream'),
  },
  dialog: {
    openFile: (options) => ipcRenderer.invoke('dialog:openFile', options),
    openDirectory: () => ipcRenderer.invoke('dialog:openDirectory'),
    saveFile: (options) => ipcRenderer.invoke('dialog:saveFile', options),
  },
  fs: {
    readFile: (path) => ipcRenderer.invoke('fs:readFile', path),
    writeFile: (path, content) => ipcRenderer.invoke('fs:writeFile', path, content),
  },
  path: {
    home: () => ipcRenderer.invoke('path:home'),
  },
}

contextBridge.exposeInMainWorld('electron', bridge)
