<template>
    <header id="header">
      <h1>EV Companion</h1>
    </header>
  </template>
  
  <script>
  export default {
    name: 'HeaderComponent',
  };
  </script>
  
  <style scoped>
  #header {
    display: flex;
    align-items: center;
    padding: 15px 20px;
    background-color: #333; /* Dark Gray */
    color: #ffa500; /* Orange */
  }
  
  #header h1 {
    margin: 0;
    font-size: 24px;
  }
  </style>
  