<template>
  <div id="app">
    <HeaderComponent /> <!-- Add your header component here -->
    <ChatComponent />
  </div>
</template>

<script>
import HeaderComponent from './components/HeaderComponent.vue'; // Import your header component

import ChatComponent from './components/ChatComponent.vue';

export default {
  name: 'App',
  components: {
    HeaderComponent, // Register your header component
    ChatComponent,
  },
};
</script>

<style>
#app {
  font-family: 'Montserrat', sans-serif;
  color: #333;
  margin-top: 20px;
}
</style>
