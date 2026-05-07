import type { ElectronBridge } from '../electron/preload'

declare global {
  interface Window {
    electron: ElectronBridge
  }
}
