<!-- Footer on every slide: current part (from the nearest earlier slide with a `part:` frontmatter) and slide counter. -->
<script setup>
import { computed } from 'vue'
import { useNav } from '@slidev/client'
const { slides, currentSlideNo, total } = useNav()
const part = computed(() => {
  for (let i = currentSlideNo.value - 1; i >= 0; i--) {
    const p = slides.value[i]?.meta?.slide?.frontmatter?.part
    if (p) return p
  }
  return ''
})
</script>
<template>
  <footer class="gw-foot">
    <span>{{ part }}</span>
    <div class="prog"><i :style="{ width: `${currentSlideNo / total * 100}%` }" /></div>
    <span>{{ currentSlideNo }} / {{ total }}</span>
  </footer>
</template>
