import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { IncidentService } from '../../services/incident.service';

interface ChatMessage {
  text: string;
  isUser: boolean;
  loading?: boolean;
}

interface AssistantResponse {
  answer: string;
  session_id?: string;
}

@Component({
  selector: 'app-assistant-chat',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatSnackBarModule
  ],
  templateUrl: './assistant-chat.html',
  styleUrl: './assistant-chat.scss'
})
export class AssistantChatComponent implements OnInit {
  userQuery = '';
  messages: ChatMessage[] = [];
  sessionId: string | null = null;
  isLoading = false;

  constructor(
    private incidentService: IncidentService,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    // Load session ID and messages from localStorage if available
    const savedSessionId = localStorage.getItem('chatSessionId');
    const savedMessages = localStorage.getItem('chatMessages');
    
    if (savedSessionId) {
      this.sessionId = savedSessionId;
    }
    
    if (savedMessages) {
      try {
        this.messages = JSON.parse(savedMessages);
      } catch (e) {
        console.error('Failed to parse saved messages', e);
        this.messages = [];
      }
    }
  }

  askQuestion(): void {
    if (this.isLoading || !this.userQuery.trim()) return;
    
    this.isLoading = true;
    
    try {
      // Add user message
      const userMessage = { text: this.userQuery, isUser: true };
      this.messages.push(userMessage);
      this.saveMessagesToStorage();
      
      // Add loading indicator
      const loadingMsg = { text: 'Thinking...', isUser: false, loading: true };
      this.messages.push(loadingMsg);
      
      // Prepare conversation history (text only)
      const history = this.messages
        .filter(m => !m.loading)
        .map(m => m.text);

      // Clear input
      this.userQuery = '';

      // Call service with sessionId
      this.incidentService.askAssistant(this.userQuery, history, this.sessionId)
        .subscribe({
          next: (response: AssistantResponse) => {
            // Remove loading message
            this.messages = this.messages.filter(m => !m.loading);
            
            // Add assistant response
            const assistantMessage = {
              text: response.answer,
              isUser: false
            };
            this.messages.push(assistantMessage);
            
            // Update sessionId if provided
            if (response.session_id) {
              this.sessionId = response.session_id;
              localStorage.setItem('chatSessionId', this.sessionId);
            }
            
            this.saveMessagesToStorage();
            this.isLoading = false;
          },
          error: (error: any) => {
            // Remove loading message
            this.messages = this.messages.filter(m => !m.loading);
            this.isLoading = false;
            
            // Show error
            const errorMsg = error.error?.message || 'Error getting response';
            this.snackBar.open(errorMsg, 'Close', {
              duration: 3000
            });
            console.error('Error:', error);
          }
        });
    } catch (error: any) {
      this.isLoading = false;
      console.error('Unexpected error:', error);
      this.snackBar.open('An unexpected error occurred', 'Close', {
        duration: 3000
      });
    }
  }

  private saveMessagesToStorage(): void {
    // Don't save loading messages
    const messagesToSave = this.messages.filter(m => !m.loading);
    localStorage.setItem('chatMessages', JSON.stringify(messagesToSave));
  }

  clearConversation(): void {
    this.messages = [];
    this.sessionId = null;
    localStorage.removeItem('chatSessionId');
    localStorage.removeItem('chatMessages');
  }
}
