import { defineShortcutsSetup } from '@slidev/types'
import { resetActive, stepActive, transport } from '../engine/gridworld.js'

// Animation keys: p play/pause · . step · r reset · + / - speed
export default defineShortcutsSetup((_nav, base) => [
  ...base,
  { name: 'gw_play', key: 'p', fn: () => { transport.playing = !transport.playing } },
  { name: 'gw_step', key: '.', fn: stepActive },
  { name: 'gw_reset', key: 'r', fn: resetActive },
  { name: 'gw_faster', key: '=', fn: () => { transport.speed *= 2 } },
  { name: 'gw_faster2', key: '+', fn: () => { transport.speed *= 2 } },
  { name: 'gw_slower', key: '-', fn: () => { transport.speed /= 2 } },
])
