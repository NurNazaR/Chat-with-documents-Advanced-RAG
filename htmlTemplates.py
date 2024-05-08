css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}

.source-container {
  border: 1px solid #ccc;
  margin-bottom: 10px;
  padding: 5px;
  border-radius: 5px;
}

.source-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
}

.source-title {
  font-weight: bold;
}

.source-pages {
  font-style: italic;
}

.source-content {
  background-color: white;
  border: 1px solid #ccc;
  padding: 10px;
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 1000;
  display: none;
  max-width: 80%;
}
'''


bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

source_template = '''
<div class="source-header"> 
  <span class="source-title">{{source_name}}</span> 
  <span class="source-pages">Pages: {{pages}}</span> 
</div>
'''
