<template>
  <div v-show="props.item">
    <h2>当前任务</h2>
    <ol>
      <audio id="notificationAudioBase" :src="MusicBase"></audio>
      <audio id="notificationAudioShort" :src="MusicShort"></audio>
      <li class="FOCUS">
        【{{ displayTimeStr }}】{{ item?.taskName }}
        <a class="function function-1" title="取消任务" @click="finishTask('undo')">
          <font-awesome-icon :icon="['fas', 'undo']" />
        </a>
        <a class="function function-0" title="完成任务" @click="finishTask('done')">
          <font-awesome-icon :icon="['fas', 'check']" />
        </a>
      </li>
    </ol>
    <!-- <LedLight :msg="displayTimeStr"></LedLight> -->
  </div>
</template>

<script setup lang="ts">
import MusicBase from './M01.mp3'
import MusicShort from './S01.mp3'
import { computed, onUnmounted, ref, watch } from 'vue'
import type { TomatoEventType, TomatoItem, TomatoParam } from './types'
import { OneMinuteMS } from './types'

// import LedLight from '@/components/led/LedLight.vue'

const props = defineProps<{
  item?: TomatoItem
}>()

const emit = defineEmits<{
  (e: 'done-task', type: TomatoEventType, param: TomatoParam): void
}>()

// RemainingSeconds 当前番茄钟剩余时间
let rs = ref(0)
// 当前计时器
let timer: number | undefined = undefined

// 番茄钟计数结束标记
let hasTomatoFinished: boolean = true

// 监听item重置行为
watch(
  () => props.item,
  () => {
    clearInterval(timer)  // 始终先清除之前可能存在的定时器, 然后再开启新的定时器
    if (props.item !== undefined) {
      timer = setInterval(updateTomato, 500)
      hasTomatoFinished = false
    }
  },
)


// 核心刷新函数, 每0.5秒刷新一次状态
function updateTomato() {
  if (!props.item) {
    return
  }

  // 更新剩余时间, 驱动页面刷新
  rs.value = calcRS(props.item)

  // 倒计时结束且当且是未结束状态, 则触发结束动作
  if (rs.value < 0 && !hasTomatoFinished) {
    // 立即标记为结束状态, 避免定时器堆积时产生多次提交操作
    hasTomatoFinished = true
    finishTask('auto')
  }
}

function finishTask(type: TomatoEventType) {
  clearInterval(timer)
  if (props.item === undefined) {
    return
  }

  emit('done-task', type, { id: props.item.itemId})
}

const displayTimeStr = computed(() => {
  if (rs.value < 0) {
    return '00:00'
  }

  const m = Math.floor(rs.value / 60)
  const s = Math.floor(rs.value % 60)

  return m.toString().padStart(2, '0') + ':' + s.toString().padStart(2, '0')
})

function calcRS(item: TomatoItem) {
  const tomatoTimeMS = 25 * OneMinuteMS
  const finishedSecond = (new Date(item.startTime).getTime()) + tomatoTimeMS
  const tsNow = new Date().getTime()
  // console.log(item.startTime, new Date(item.startTime))
  return (finishedSecond - tsNow) / 1000
}

// 页面卸载时强制清理定时器
// 如果当前页面正在进行倒计时, 直接push切换页面后, 定时器并不会自动销毁
// 因此不手动清理, 后续用户再次打开番茄钟页面, 可能会产生两个定时器进行倒计时, 进而导致两次提交
onUnmounted(() => {
  clearInterval(timer)
})

</script>

<style scoped>
/*清除ol和ul标签的默认样式*/
ol,
ul {
  padding: 0;
  list-style: none;
}

.FOCUS {
  border-left: 5px solid #ee0000;
  line-height: 32px;
  background: #fff;
  position: relative;
  margin-bottom: 10px;
  padding: 0 88px 0 8px;
  border-radius: 3px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.07);
}

.function {
  position: absolute;
  display: inline-block;
  width: 14px;
  height: 12px;
  line-height: 14px;
  text-align: center;
  color: #888;
  font-weight: bold;
  font-size: 20px;
  cursor: pointer;
}

.function-0 {
  top: 6px;
  right: 14px;
}

.function-1 {
  top: 6px;
  right: 44px;
}
</style>
