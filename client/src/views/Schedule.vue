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
            Id: 1,
            Subject: "值班：A",
            StartTime: new Date(2024,11,2,9,0),
            EndTime: new Date(2024,11,2,17,0),
            IsAllDay: false
          },
          {
            Id: 2,
            Subject: "值班：A",
            StartTime: new Date(2024,11,39,0 ,0),
            EndTime: new Date(2024,11,3,17,0),
            IsAllDay: false
          },
          {
            Id: 3,
            Subject: "值班：A",
            StartTime: new Date(2024,11,4,9,0),
            EndTime: new Date(2024,11,4,17,0),
            IsAllDay: false
          },
          {
            Id: 4,
            Subject: "值班：A",
            StartTime: new Date(2024,11,5,9,0),
            EndTime: new Date(2024,11,5,17,0),
            IsAllDay: false
          },
          {
            Id: 5,
            Subject: "值班：A",
            StartTime: new Date(2024,11,6,9,0),
            EndTime: new Date(2024,11,6,17,0),
            IsAllDay: false
          },
          {
            Id: 6,
            Subject: "值班：A",
            StartTime: new Date(2024,11,9,9,0),
            EndTime: new Date(2024,11,9,17,0),
            IsAllDay: false
          },
          {
            Id: 7,
            Subject: "值班：A",
            StartTime: new Date(2024,11,11,9,0),
            EndTime: new Date(2024,11,11,17,0),
            IsAllDay: false
          },
          {
            Id: 8,
            Subject: "值班：A",
            StartTime: new Date(2024,11,13,9,0),
            EndTime: new Date(2024,11,13,17,0),
            IsAllDay: false
          },
          {
            Id: 9,
            Subject: "值班：A",
            StartTime: new Date(2024,11,17,9,0),
            EndTime: new Date(2024,11,17,17,0),
            IsAllDay: false
          },
          {
            Id: 10,
            Subject: "值班：A",
            StartTime: new Date(2024,11,19,9,0),
            EndTime: new Date(2024,11,19,17,0),
            IsAllDay: false
          },
          {
            Id: 11,
            Subject: "值班：A",
            StartTime: new Date(2024,11,23,9,0),
            EndTime: new Date(2024,11,23,17,0),
            IsAllDay: false
          },
          {
            Id: 12,
            Subject: "值班：A",
            StartTime: new Date(2024,11,25,9,0),
            EndTime: new Date(2024,11,25,17,0),
            IsAllDay: false
          },
          {
            Id: 13,
            Subject: "值班：A",
            StartTime: new Date(2024,11,27,9,0),
            EndTime: new Date(2024,11,27,17,0),
            IsAllDay: false
          },
          {
            Id: 14,
            Subject: "值班：A",
            StartTime: new Date(2024,11,31,9,0),
            EndTime: new Date(2024,11,3117,0,0),
            IsAllDay: false
          },
          {
            Id: 15,
            Subject: "值班：B",
            StartTime: new Date(2024,11,3,7,0),
            EndTime: new Date(2024,11,3,15,0),
            IsAllDay: false
          },
          {
            Id: 16,
            Subject: "值班：B",
            StartTime: new Date(2024,11,7,9,0),
            EndTime: new Date(2024,11,7,15,0),
            IsAllDay: false
          },
          {
            Id: 17,
            Subject: "值班：B",
            StartTime: new Date(2024,11,13,9,0),
            EndTime: new Date(2024,11,13,15,0),
            IsAllDay: false
          },
          {
            Id: 18,
            Subject: "值班：B",
            StartTime: new Date(2024,11,20,9,0),
            EndTime: new Date(2024,11,20,15,0),
            IsAllDay: false
          },
          {
            Id: 19,
            Subject: "值班：B",
            StartTime: new Date(2024,11,22,17,0),
            EndTime: new Date(2024,11,22,23,0),
            IsAllDay: false
          },
          {
            Id: 20,
            Subject: "值班：B",
            StartTime: new Date(2024,11,27,9,0),
            EndTime: new Date(2024,11,27,15,0),
            IsAllDay: false
          },
          {
            Id: 21,
            Subject: "值班：B",
            StartTime: new Date(2024,11,29,15,0),
            EndTime: new Date(2024,11,29,23,0),
            IsAllDay: false
          },
          {
            Id: 22,
            Subject: "值班：C",
            StartTime: new Date(2024,11,4,9,0),
            EndTime: new Date(2024,11,4,15,0),
            IsAllDay: false
          },
          {
            Id: 23,
            Subject: "值班：©",
            StartTime: new Date(2024,11,8,9,0),
            EndTime: new Date(2024,11,8,15,0),
            IsAllDay: false
          },
          {
            Id: 24,
            Subject: "值班：©",
            StartTime: new Date(2024,11,11,9,0),
            EndTime: new Date(2024,11,11,15,0),
            IsAllDay: false
          },
          {
            Id: 25,
            Subject: "值班：C",
            StartTime: new Date(2024,11,14,15,0),
            EndTime: new Date(2024,11,14,23,0),
            IsAllDay: false
          },
          {
            Id: 26,
            Subject: "值班：|",
            StartTime: new Date(2024,11,18,9,0),
            EndTime: new Date(2024,11,18,15,0),
            IsAllDay: false
          },
          {
            Id: 27,
            Subject: "值班：©",
            StartTime: new Date(2024,11,21,15,0),
            EndTime: new Date(2024,11,21,23,0),
            IsAllDay: false
          },
          {
            Id: 28,
            Subject: "值班：|",
            StartTime: new Date(2024,11,25,9,0),
            EndTime: new Date(2024,11,25,15,0),
            IsAllDay: false
          },
          {
            Id: 29,
            Subject: "值班：|",
            StartTime: new Date(2024,11,28,15,0),
            EndTime: new Date(2024,11,28,23,0),
            IsAllDay: false
          },
          {
            Id: 30,
            Subject: "值班：©",
            StartTime: new Date(2024,11,30,7,0),
            EndTime: new Date(2024,11,30,15,0),
            IsAllDay: false
          },
          {
            Id: 31,
            Subject: "值班：D",
            StartTime: new Date(2024,11,6,160,0),
            EndTime: new Date(2024,11,6,23,0),
            IsAllDay: false
          },
          {
            Id: 32,
            Subject: "值班：D",
            StartTime: new Date(2024,11,8,15,0),
            EndTime: new Date(2024,11,8,23,0),
            IsAllDay: false
          },
          {
            Id: 33,
            Subject: "值班：D",
            StartTime: new Date(2024,11,10,16,0),
            EndTime: new Date(2024,11,10,23,0),
            IsAllDay: false
          },
          {
            Id: 34,
            Subject: "值班：D",
            StartTime: new Date(2024,11,13,16,0),
            EndTime: new Date(2024,11,13,23,0),
            IsAllDay: false
          },
          {
            Id: 35,
            Subject: "值班：D",
            StartTime: new Date(2024,11,17,16,0),
            EndTime: new Date(2024,11,17,230,0),
            IsAllDay: false
          },
          {
            Id: 36,
            Subject: "值班：D",
            StartTime: new Date(2024,11,20,16,0),
            EndTime: new Date(2024,11,20,23,0),
            IsAllDay: false
          },
          {
            Id: 37,
            Subject: "值班：D",
            StartTime: new Date(2024,11,24,16,0),
            EndTime: new Date(2024,11,24,23,0),
            IsAllDay: false
          },
          {
            Id: 38,
            Subject: "值班：D",
            StartTime: new Date(2024,11,27,16,0),
            EndTime: new Date(2024,11,27,23,0),
            IsAllDay: false
          },
          {
            Id: 39,
            Subject: "值班：D",
            StartTime: new Date(2024,11,31,16,0),
            EndTime: new Date(2024,11,31,23,0),
            IsAllDay: false
          },
          {
            Id: 40,
            Subject: "值班：E",
            StartTime: new Date(2024,11,4,16,0),
            EndTime: new Date(2024,11,4,23,0),
            IsAllDay: false
          },
          {
            Id: 41,
            Subject: "值班：E",
            StartTime: new Date(2024,11,12,16,0),
            EndTime: new Date(2024,11,12,23,0),
            IsAllDay: false
          },
          {
            Id: 42,
            Subject: "值班：E",
            StartTime: new Date(2024,11,15,7,0),
            EndTime: new Date(2024,11,15,15,0),
            IsAllDay: false
          },
          {
            Id: 43,
            Subject: "值班：E",
            StartTime: new Date(2024,11,1916,0,0),
            EndTime: new Date(2024,11,19,23,0),
            IsAllDay: false
          },
          {
            Id: 44,
            Subject: "值班：|",
            StartTime: new Date(2024,11,22,9,0),
            EndTime: new Date(2024,11,22,17,0),
            IsAllDay: false
          },
          {
            Id: 45,
            Subject: "值班：E",
            StartTime: new Date(2024,11,26,16,0),
            EndTime: new Date(2024,11,26,23,0),
            IsAllDay: false
          },
          {
            Id: 46,
            Subject: "值班：F",
            StartTime: new Date(2024,11,3,23,0),
            EndTime: new Date(2024,11,3,9,0),
            IsAllDay: false
          },
          {
            Id: 47,
            Subject: "值班：F",
            StartTime: new Date(2024,11,4,23,0),
            EndTime: new Date(2024,11,4,9,0),
            IsAllDay: false
          },
          {
            Id: 48,
            Subject: "值班：F",
            StartTime: new Date(2024,11,6,23,0),
            EndTime: new Date(2024,11,6,9,0),
            IsAllDay: false
          },
          {
            Id: 49,
            Subject: "值班：F",
            StartTime: new Date(2024,11,7,23,0),
            EndTime: new Date(2024,11,7,9,0),
            IsAllDay: false
          },
          {
            Id: 50,
            Subject: "值班：F",
            StartTime: new Date(2024,11,10,23,0),
            EndTime: new Date(2024,11,10,9,0),
            IsAllDay: false
          },
          {
            Id: 51,
            Subject: "值班：|",
            StartTime: new Date(2024,11,11,23,0),
            EndTime: new Date(2024,11,11,9,0),
            IsAllDay: false
          },
          {
            Id: 52,
            Subject: "值班：F",
            StartTime: new Date(2024,11,12,23,0),
            EndTime: new Date(2024,11,12,9,0),
            IsAllDay: false
          },
          {
            Id: 53,
            Subject: "值班：F",
            StartTime: new Date(2024,11,18,23,0),
            EndTime: new Date(2024,11,18,9,0),
            IsAllDay: false
          },
          {
            Id: 54,
            Subject: "值班：F",
            StartTime: new Date(2024,11,20,23,0),
            EndTime: new Date(2024,11,20,7,0),
            IsAllDay: false
          },
          {
            Id: 55,
            Subject: "值班：F",
            StartTime: new Date(2024,11,26,23,0),
            EndTime: new Date(2024,11,26,9,0),
            IsAllDay: false
          },
          {
            Id: 56,
            Subject: "值班：F",
            StartTime: new Date(2024,11,28,23,0),
            EndTime: new Date(2024,11,28,9,0),
            IsAllDay: false
          },
          {
            Id: 57,
            Subject: "值班：F",
            StartTime: new Date(2024,11,31,23,0),
            EndTime: new Date(2024,11,31,7,0),
            IsAllDay: false
          },
          {
            Id: 58,
            Subject: "值班：G",
            StartTime: new Date(2024,11,1,23,0),
            EndTime: new Date(2024,11,1,7,0),
            IsAllDay: false
          },
          {
            Id: 59,
            Subject: "值班：G",
            StartTime: new Date(2024,11,9,23,0),
            EndTime: new Date(2024,11,9,7,0),
            IsAllDay: false
          },
          {
            Id: 60,
            Subject: "值班：G",
            StartTime: new Date(2024,11,15,23,0),
            EndTime: new Date(2024,11,15,7,0),
            IsAllDay: false
          },
          {
            Id: 61,
            Subject: "值班：G",
            StartTime: new Date(2024,11,16,23,0),
            EndTime: new Date(2024,11,16,7,0),
            IsAllDay: false
          },
          {
            Id: 62,
            Subject: "值班：G",
            StartTime: new Date(2024,11,22,23,0),
            EndTime: new Date(2024,11,22,7,0),
            IsAllDay: false
          },
          {
            Id: 63,
            Subject: "值班：G",
            StartTime: new Date(2024,11,23,25,0),
            EndTime: new Date(2024,11,23,7,0),
            IsAllDay: false
          },
          {
            Id: 64,
            Subject: "值班：G",
            StartTime: new Date(2024,11,29,23,0),
            EndTime: new Date(2024,11,29,7,0),
            IsAllDay: false
          },
          {
            Id: 65,
            Subject: "值班：G",
            StartTime: new Date(2024,11,30,23,0),
            EndTime: new Date(2024,11,30,7,0),
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