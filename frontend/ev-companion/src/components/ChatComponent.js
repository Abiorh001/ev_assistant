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
  