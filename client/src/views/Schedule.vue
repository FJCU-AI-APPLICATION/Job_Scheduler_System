<template>
  <div class="timesheet-wrapper">
    <h1>班表查詢</h1>
    <DayPilotScheduler :config="config" ref="schedulerRef" />
  </div>
</template>

<script setup>
import { DayPilot, DayPilotScheduler } from "daypilot-pro-vue";
import { reactive, ref, onMounted } from "vue";
import { eventList } from "@/data/eventData.js";

const schedulerRef = ref(null);

const config = reactive({
  locale: "zh-tw",
  rowHeaderColumns: [
    {
      title: "Date",
      width: 81,
    },
    {
      title: "Total",
      width: 80,
    },
  ],

  onBeforeRowHeaderRender: (args) => {
    const totalHours = args.row.events.totalDuration().totalHours();
      if (totalHours > 0) {
        args.row.columns[1].text = `${totalHours.toFixed(0)} hrs`;
      }
    },
  cellWidthSpec: "Auto",
  cellWidthMin: 20,
  crosshairType: "Header",
  autoScroll: "Drag",
  timeHeaders: [
    {
      groupBy: "Hour",
    },
  ],
  scale: "Hour",
  days: DayPilot.Date.today().daysInMonth(),
  viewType: "Days",
  startDate: DayPilot.Date.today().firstDayOfMonth(),
  showNonBusiness: true,
  businessBeginsHour: 0,
  businessEndsHour: 24,
  businessWeekends: true,
  floatingEvents: true,
  eventHeight: 80,
  eventMovingStartEndEnabled: false,
  eventResizingStartEndEnabled: false,
  timeRangeSelectingStartEndEnabled: false,
  groupConcurrentEvents: false,
  eventStackingLineHeight: 100,
  allowEventOverlap: true,
  timeRangeSelectedHandling: "Enabled",

  onBeforeEventRender: args => {
    const shift = args.data.shiftType || "（未指定）";
    const empList = Array.isArray(args.data.employees) ? args.data.employees.join("、") : "（無）";
    const headCount = Array.isArray(args.data.employees) ? `${args.data.employees.length}人` : "（未知）";

    let bgColor, bgColor2, bgColor3;
    switch (args.data.shiftType) {
      case "晚班":
        bgColor  = "#ce7c63";
        bgColor2 = "#d7886b";
        bgColor3 = "#e2a077";
        break;
      case "大夜班":
        bgColor  = "#2c3e50";
        bgColor2 = "#3b4d5d";
        bgColor3 = "#4f6370";
        break;
      default:
        bgColor  = "#0ca5b6";
        bgColor2 = "#26b2c2";
        bgColor3 = "#40bfce";
    }

    args.data.areas = [
      { top: 0, left: 0, right: 0, height: 6, backColor: bgColor },

      { left: 6, top: 8, width: 14, height: 14, backColor: bgColor, style: "border-radius: 2px;" },
      { left: 26, top: 8, text: `班別：${shift}`, fontColor: "#2c3e50" },

      { left: 6, top: 32, width: 14, height: 14, backColor: bgColor2, style: "border-radius: 2px;" },
      { left: 26, top: 32, text: `員工：${empList}`, fontColor: "#2c3e50" },

      { left: 6, top: 56, width: 14, height: 14, backColor: bgColor3, style: "border-radius: 2px;" },
      { left: 26, top: 56, text: `人數：${headCount}`, fontColor: "#2c3e50" }
    ];
  },

  onTimeRangeSelected: async (args) => {
    const timesheet = args.control;
    const modal = await DayPilot.Modal.prompt("Create a new event:", "Event 1");
    timesheet.clearSelection();
    if (modal.canceled) { return; }
    timesheet.events.add({
      start: args.start,
      end: args.end,
      id: DayPilot.guid(),
      resource: args.resource,
      text: modal.result
    });
  },
  eventMoveHandling: "Update",
  onEventMoved: (args) => {
    args.control.message("Event moved: " + args.e.text());
  },
  eventResizeHandling: "Update",
  onEventResized: (args) => {
    args.control.message("Event resized: " + args.e.text());
  },
  eventDeleteHandling: "Update",
  onEventDeleted: (args) => {
    args.control.message("Event deleted: " + args.e.text());
  },
  eventClickHandling: "Disabled",
  eventHoverHandling: "Disabled",
});

onMounted(() => {
  // config.events = eventList;
  console.log("載入事件：", eventList);
  if (schedulerRef.value && schedulerRef.value.control) {
    schedulerRef.value.control.update({ events: eventList });  // ✅ 主動刷新顯示
    schedulerRef.value.control.scrollTo("2025-06-01T08:00:00");
  }
});
</script>

<style>
.timesheet-wrapper {
  width: 100%;
  min-width: 900px;
  overflow-x: auto;
  /* border: 1px solid #ccc; */
}

/* .scheduler_default_columnheader,
.scheduler_default_rowheader {
  text-align: center;
} */

.scheduler_default_timeheadercol_inner {
  /* font-size: 13px; */
  text-align: center;  
  line-height: 1;
  white-space: normal;
}

</style>