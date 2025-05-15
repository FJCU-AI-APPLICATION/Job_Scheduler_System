<template>
  <div>
    <ejs-schedule
      :height="'550px'"
      :selectedDate="selectedDate"
      :eventSettings="eventSettings"
      :currentView="'Month'">
    </ejs-schedule>
  </div>
</template>

<script>
import {
  ScheduleComponent,
  Day,
  Week,
  WorkWeek,
  Month,
  Agenda,
  Resize,
  DragAndDrop
} from '@syncfusion/ej2-vue-schedule';

export default {
  name: "SchedulePage",
  props: {
    scheduleData: {
        type: Array,
        required: true
    }
  },
  data() {
    return {
      selectedDate: new Date(2025, 4, 20), // 預設日期：2025/5/20，注意1月份從0起算
      eventSettings: {
        // dataScource: this.scheduleData
        dataSource: [
          {
            Id: 1,
            Subject: "早班：郭小姐",
            StartTime: new Date(2025, 4, 20, 9, 0),
            EndTime: new Date(2025, 4, 20, 13, 0),
            IsAllDay: false
          },
          {
            Id: 2,
            Subject: "晚班：黃先生",
            StartTime: new Date(2025, 4, 20, 14, 0),
            EndTime: new Date(2025, 4, 20, 18, 0),
            IsAllDay: false
          }
        ]
      }
    };
  },

  provide: {
    schedule: [Day, Week, WorkWeek, Month, Agenda, Resize, DragAndDrop]
  }
};
</script>