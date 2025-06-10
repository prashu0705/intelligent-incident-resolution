import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
// FormsModule is not directly used by AppComponent anymore, but AssistantChatComponent might import it itself.
// If AssistantChatComponent is standalone and imports FormsModule, this import here might be redundant.
// However, keeping it doesn't harm if other future direct uses in AppComponent template arise.
import { FormsModule } from '@angular/forms';
import { AssistantChatComponent } from './components/assistant-chat/assistant-chat';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule, // Keep for now, AssistantChatComponent is standalone and should import what it needs.
    AssistantChatComponent // Import the AssistantChatComponent
  ],
  templateUrl: './app.html',
  styleUrls: ['./app.scss'],
})
export class AppComponent {
  // All old chat-related properties and methods are removed.
  // AppComponent might have other application-wide logic or properties in a larger app.
  // For this subtask, it becomes very minimal.
}

