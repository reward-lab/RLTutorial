<!--
  Two-column slide: kicker + title on top, talking points (::left:: slot) and a live readout on the left,
  the gridworld animation for `scene` on the right with its controls and a small transport bar.
  Frontmatter: scene (engine id, required), kicker, cover (true = title slide, default slot centred on the left).
-->
<script setup>
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import { useNav, useSlideContext } from '@slidev/client'
import { activate, bindCanvas, getGeo, getScene, isTerm, isWall, readTokens, R, C, transport } from '../engine/gridworld.js'

const props = defineProps({ scene: { type: String, required: true }, kicker: String, cover: Boolean, paused: Boolean })
const { $page, $renderContext, $scale } = useSlideContext()
const { currentSlideNo } = useNav()
const sc = getScene(props.scene)

const live = computed(() => ['slide', 'presenter'].includes($renderContext.value))
const current = computed(() => live.value && currentSlideNo.value === $page.value)

const stage = ref(), cv = ref(), pc = ref()
const liveHtml = ref(''), ver = ref(0), isPlaying = ref(transport.playing), speed = ref(transport.speed)
const controls = shallowRef([])
const refreshControls = () => { ver.value++; controls.value = sc.controls ? sc.controls() : [] }
let size = { w: 0, h: 0 }, raf = 0, last = 0, ro

function enter() {
  activate(sc, $page.value)
  if (props.paused) { transport.playing = false; isPlaying.value = false} else {transport.playing = true; isPlaying.value = true}
}
function fitCanvas(c, w, h) {
  const d = (window.devicePixelRatio || 1) * ($scale?.value || 1), ctx = c.getContext('2d')
  if (c.width !== Math.round(w * d)) { c.width = Math.round(w * d); c.height = Math.round(h * d) }
  ctx.setTransform(d, 0, 0, d, 0, 0)
  return ctx
}
function draw(dt) {
  if (!cv.value || !size.w) return
  readTokens()
  bindCanvas(fitCanvas(cv.value, size.w, size.h), size.w, size.h)
  sc.draw(dt)
  if (sc.panelCanvas && pc.value) { const w = pc.value.clientWidth, h = pc.value.clientHeight; sc.drawPanel(fitCanvas(pc.value, w, h), w, h) }
  sc.dom && sc.dom()
  const html = sc.live ? sc.live() : ''
  if (html !== liveHtml.value) liveHtml.value = html
}
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now
  sc.update && sc.update(dt)
  draw(dt)
  isPlaying.value = transport.playing; speed.value = transport.speed
  raf = current.value ? requestAnimationFrame(frame) : 0
}
function start() { if (!raf) { last = performance.now(); raf = requestAnimationFrame(frame) } }

watch(current, (on) => { if (on) { enter(); refreshControls(); start() } })
onMounted(() => {
  ro = new ResizeObserver(() => { size = { w: stage.value.clientWidth, h: stage.value.clientHeight }; draw(0) })
  ro.observe(stage.value)
  refreshControls()
  if (current.value) { enter(); start() }
})
onBeforeUnmount(() => { cancelAnimationFrame(raf); raf = 0; ro && ro.disconnect() })

function setCtl(c, v) { c.set(v); refreshControls(); if (!current.value) draw(0) }
function onClick(e) {
  if (!sc.click) return
  const b = cv.value.getBoundingClientRect(), k = size.w / b.width, { ox, oy, cs } = getGeo()
  const c = Math.floor(((e.clientX - b.left) * k - ox) / cs), r = Math.floor(((e.clientY - b.top) * k - oy) / cs)
  if (r < 0 || r >= R || c < 0 || c >= C) return
  const s = r * C + c
  if (!isWall(s) && !isTerm(s)) sc.click(s)
}
const togglePlay = () => { transport.playing = !transport.playing; isPlaying.value = transport.playing }
const doStep = () => sc.step && sc.step()
const doReset = () => { enter(); refreshControls() }
const setSpeed = (f) => { transport.speed = transport.speed * f; speed.value = transport.speed }
const label = l => String(l).replace(/\^π/g, '<sup>π</sup>')
</script>

<template>
  <div class="slidev-layout demo" :class="{ cover }">
    <header v-if="!cover" class="head">
      <div v-if="kicker" class="kicker">{{ kicker }}</div>
      <slot />
    </header>
    <div class="body">
      <div class="left">
        <slot v-if="cover" />
        <slot name="left" />
        <div v-if="sc.extra" class="extra" v-html="sc.extra" />
        <canvas v-if="sc.panelCanvas" ref="pc" class="pc" />
        <div v-if="liveHtml" class="live" v-html="liveHtml" />
      </div>
      <div class="right">
        <div ref="stage" class="cvwrap">
          <canvas ref="cv" :style="{ cursor: sc.click ? 'pointer' : 'default' }" aria-label="Gridworld animation" @click="onClick" />
        </div>
        <div class="controls" :data-v="ver">
          <template v-for="(c, i) in controls" :key="i + c.label">
            <div v-if="c.type === 'seg'" class="ctl">
              <span class="lab">{{ c.label }}</span>
              <div class="seg" role="group">
                <button v-for="[v, l] in c.opts" :key="v" :aria-pressed="c.get() === v" @click="setCtl(c, v)" v-html="label(l)" />
              </div>
            </div>
            <div v-else class="ctl">
              <label>{{ c.label }}<output>{{ c.fmt(c.get()) }}</output></label>
              <input type="range" :min="c.min" :max="c.max" :step="c.step" :value="c.get()" @input="setCtl(c, +$event.target.value)">
            </div>
          </template>
          <div v-if="sc.step || sc.update" class="tp" role="group" aria-label="Animation">
            <button :title="isPlaying ? 'Pause (p)' : 'Play (p)'" @click="togglePlay">{{ isPlaying ? '❚❚' : '▶' }}</button>
            <button title="Step (.)" @click="doStep">⇥</button>
            <button title="Reset (r)" @click="doReset">↺</button>
            <button title="Slower (-)" @click="setSpeed(0.5)">−</button>
            <span class="speed">{{ speed }}×</span>
            <button title="Faster (+)" @click="setSpeed(2)">+</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
