<!-- Kicker + title on top, talking points on the left (::left::), a code block on the right (::right::). -->
<script setup>
defineProps({ kicker: String })
</script>

<template>
  <div class="slidev-layout demo codeslide">
    <header class="head">
      <div v-if="kicker" class="kicker">{{ kicker }}</div>
      <slot />
    </header>
    <div class="body">
      <div class="left"><slot name="left" /></div>
      <div class="right code"><slot name="right" /></div>
    </div>
  </div>
</template>
