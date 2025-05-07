<template>
    <div>
        <transition name="modal">
            <div class="modal-mask">
                <div class="modal-wrapper">
                    <div class="modal-container">

                    <div class="modal-header">
                        <slot name="header">
                        default header
                        </slot>
                    </div>

                    <div class="modal-body">
                        <slot name="body">
                        default body
                        </slot>
                    </div>

                    <div class="modal-footer">
                        <slot name="footer">
                        default footer
                        <button class="modal-default-button" @click="$emit('close')">
                            OK
                        </button>
                        </slot>
                    </div>
                    </div>
                </div>
            </div>
        </transition>
    </div>
    <!-- <div>
        <b-button @click="show = true">打開 Modal</b-button>

        <b-modal
            v-model="show"
            title="測試 Modal"
            :modal-class="'modal-zfix'"
            @shown="raiseModalZIndex"
        >
            
            <p>我是內容</p>
        </b-modal>
        
    </div> -->
</template>
  

<script>
    export default {
        name: "TestModal",
        data() {
            return {
            showModal: false
            };
        }
    };
</script>

<style>
.modal-mask {
  position: fixed;
  z-index: 9998;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: table;
  transition: opacity 0.3s ease;
}

.modal-wrapper {
  display: table-cell;
  vertical-align: middle;
}

.modal-container {
  width: 300px;
  margin: 0 auto;
  padding: 20px 30px;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.33);
  font-family: Helvetica, Arial, sans-serif;
  transition: all 0.3s ease;
}

.modal-header h3 {
  margin-top: 0;
  color: #42b983;
}

.modal-footer {
  margin-top: 20px;
  text-align: right;
}

.modal-default-button {
  background-color: #42b983;
  border: none;
  padding: 6px 12px;
  color: #fff;
  border-radius: 4px;
  cursor: pointer;
}

/* 動畫 */
.modal-enter {
  opacity: 0;
}
.modal-enter-active {
  opacity: 1;
  transition: opacity 0.3s ease;
}
.modal-leave-active {
  opacity: 0;
  transition: opacity 0.3s ease;
}
</style>


<!-- 
<script>
    export default {
        name: "TestModal",
        data() {
            return { show: false };
        },
        methods: {
            raiseModalZIndex() {
                document.body.classList.remove('modal-open');
                document.body.style.overflow = 'visible'; // ← 解鎖畫面空間
                this.$nextTick(() => {
                    const modalOuter = document.querySelector('[id^="__BVID__"][id$="__BV_modal_outer_"]');
                    if (modalOuter) {
                        modalOuter.style.zIndex = 9999;
                    }
                });
            }
        }
    }
</script>
   -->
<!-- 
<style>
.modal.show {
    overflow: visible ;
}
.modal-zfix {
    z-index: 1060 !important;
    overflow: visible !important;
}

.modal-backdrop {
    z-index: 1050 !important;
}

.body.modal-open {
  overflow: visible !important;
}
</style> -->