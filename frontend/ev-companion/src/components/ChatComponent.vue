<template>
  <div id="container">
    <main id="conversation" aria-live="polite">
      <div v-for="message in messages" :key="message.id" :class="message.type">
        <p v-html="message.content"></p>
      </div>
    </main>
    <div id="suggestions">
      <button v-for="suggestion in suggestions" :key="suggestion" class="suggestion-btn" @click="sendSuggestion(suggestion)">
        {{ suggestion }}
      </button>
    </div>
    <footer id="input-container">
      <textarea v-model="messageInput" @keydown.enter="sendMessage" placeholder="Type your message..." aria-label="Message input"></textarea>
      <button id="send-button" @click="sendMessage" aria-label="Send message">
        <span v-if="!loading">Send</span>
        <div v-else class="loading-indicator"></div>
      </button>
    </footer>
  </div>
</template>

<script>
export default {
  data() {
    return {
      messageInput: "",
      messages: [],
      suggestions: [
        "Plan a new Trip",
        "Charge EV Anywhere",
        "Find Charging Stations",
        "Support for EV Charging",
        "Open Charge Point Protocol",
      ],
      loading: false,
      clientId: this.generateUUID(),
      webSocket: null,
    };
  },
  methods: {
    generateUUID() {
      return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
      });
    },
    connectWebSocket() {
      this.webSocket = new WebSocket(`ws://127.0.0.1:8765/ws/${this.clientId}`);

      this.webSocket.onopen = () => {
        console.log("WebSocket connection established.");
      };

      this.webSocket.onclose = () => {
        console.log("WebSocket connection closed.");
      };

      this.webSocket.onmessage = (event) => {
        this.loading = false;
        this.messages.push({ id: this.generateUUID(), type: 'response', content: this.formatResponse(event.data) });
        this.$nextTick(() => {
          const conversationDiv = this.$el.querySelector("#conversation");
          conversationDiv.scrollTop = conversationDiv.scrollHeight;
        });
      };

      this.webSocket.onerror = (event) => {
        console.error("WebSocket error:", event);
      };

      window.addEventListener("beforeunload", () => {
        this.webSocket.close();
      });
    },
    sendMessage() {
      if (this.messageInput.trim() !== "") {
        this.messages.push({ id: this.generateUUID(), type: 'message', content: this.messageInput });
        this.webSocket.send(this.messageInput);
        this.messageInput = "";
        this.loading = true;
      }
    },
    sendSuggestion(suggestion) {
      this.messages.push({ id: this.generateUUID(), type: 'message', content: suggestion });
      this.webSocket.send(suggestion);
      this.loading = true;
    },
    formatResponse(response) {
      const patterns = {
        headings: /^#+\s(.+)/,
        bold: /\*\*(.*?)\*\*/g,
        italic: /_(.*?)_/g,
        strikethrough: /~~(.*?)~~/g,
        links: /\[([^\]]+)\]\(([^)]+)\)/g,
        unorderedList: /^\s*-\s(.+)/,
        orderedList: /^\s*\d+\.\s(.+)/,
        code: /`([^`]+)`/g,
      };

      function replaceMarkdown(text) {
        text = text.replace(patterns.headings, '<h3 style="color: #ffa500;">$1</h3>');
        text = text.replace(patterns.bold, '<strong>$1</strong>');
        text = text.replace(patterns.italic, '<em>$1</em>');
        text = text.replace(patterns.strikethrough, '<del>$1</del>');
        text = text.replace(patterns.links, '<a href="$2" style="color: #ffa500;">$1</a>');
        text = text.replace(patterns.unorderedList, '<li>$1</li>');
        text = text.replace(patterns.orderedList, '<li>$1</li>');
        text = text.replace(patterns.code, '<code>$1</code>');
        return text;
      }

      const lines = response.split('\n');
      let formattedResponse = '';

      for (const line of lines) {
        formattedResponse += `<p>${replaceMarkdown(line)}</p>`;
      }
      return formattedResponse;
    }
  },
  mounted() {
    this.connectWebSocket();
  }
};
</script>

<style scoped>
body {
  font-family: 'Montserrat', sans-serif;
  margin: 0;
  padding: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background: linear-gradient(45deg, #121212, #333);
  color: #fff;
}

#container {
  display: flex;
  flex-direction: column;
  width: 800px;
  max-width: 90%;
  height: 90vh;
  background-color: #1c1c1c;
  border-radius: 10px;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
  overflow: hidden;
}

#header {
  display: flex;
  align-items: center;
  padding: 15px 20px;
  background-color: #333;
  color: #ffa500;
}

#header img {
  margin-right: 10px;
}

#header h1 {
  margin: 0;
  font-size: 24px;
}

#conversation {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  background-color: #252525;
  display: flex;
  flex-direction: column;
}

.message, .response {
  padding: 10px 15px;
  border-radius: 20px;
  margin-bottom: 10px;
  max-width: 70%;
  word-wrap: break-word;
  line-height: 1.5;
}

.message {
  background-color: #ffa500;
  color: #000;
  align-self: flex-end;
}

.response {
  background-color: #333;
  color: #fff;
  align-self: flex-start;
}

#suggestions {
  padding: 10px 20px;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  background-color: #1c1c1c;
}

.suggestion-btn {
  background-color: #ffa500;
  color: #000;
  border: none;
  padding: 10px 20px;
  margin: 5px;
  border-radius: 20px;
  cursor: pointer;
  font-size: 14px;
}

.suggestion-btn:hover {
  background-color: #ff8c00;
}

#input-container {
  display: flex;
  padding: 10px 20px;
  background-color: #1c1c1c;
  border-top: 1px solid #333;
}

#input-container textarea {
  flex: 1;
  padding: 10px;
  border: 1px solid #333;
  border-radius: 20px;
  background-color: #252525;
  color: #fff;
  margin-right: 10px;
  resize: none;
  height: 40px;
  overflow: hidden;
}

#input-container textarea:focus {
  height: auto;
  overflow: auto;
}

#send-button {
  width: 80px;
  height: 40px;
  background-color: #ffa500;
  color: #000;
  border: none;
  border-radius: 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

#send-button:hover {
  background-color: #ff8c00;
}

.loading-indicator {
  border: 4px solid #f3f3f3;
  border-top: 4px solid #333;
  border-radius: 50%;
  width: 20px;
  height: 20px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
