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
import { MatListModule } from '@angular/material/list';

interface ChatMessage {
  text: string;
  isUser: boolean;
  type: 'text' | 'loading' | 'similar_incidents' | 'recommended_resolutions';
  data?: any; // To store structured data for similar_incidents or recommended_resolutions
}

// Define more specific types for structured data
interface SimilarIncidentData {
  incident: {
    'Ticket Number': string;
    cleaned_text?: string; // Assuming this is available, or summary/combined_text
    Summary?: string;
    combined_text?: string;
    // Add other relevant fields from the incident object if needed
  };
  distance: number;
}

interface RecommendedResolutionData {
  ticket: string;
  suggested_resolution: string;
  source_of_resolution: string;
}


interface AssistantResponse {
  answer: string;
  session_id?: string;
  supporting_data?: {
    similar_incidents?: {
      results: SimilarIncidentData[];
    };
    recommend_resolutions?: {
      recommendations: RecommendedResolutionData[];
    };
    // other potential supporting data fields
  };
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
    MatSnackBarModule,
    MatListModule
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
      const userMessage: ChatMessage = { text: this.userQuery, isUser: true, type: 'text' };
      this.messages.push(userMessage);
      // No saveMessagesToStorage() here yet, save after all messages for this turn are added
      
      // Add loading indicator
      const loadingMsg: ChatMessage = { text: 'Thinking...', isUser: false, type: 'loading' };
      this.messages.push(loadingMsg);
      
      // Prepare conversation history in the new format
      const historyPayload = this.messages
        .filter(m => m.type !== 'loading' && m !== userMessage && m !== loadingMsg) // Exclude current user message and loading
        .map(m => ({
          role: m.isUser ? 'user' : 'assistant',
          content: m.text
        }));

      const currentQueryText = userMessage.text; // Store before clearing input

      // Clear input
      this.userQuery = '';

      // Call service with sessionId and the current query
      this.incidentService.askAssistant(currentQueryText, historyPayload, this.sessionId)
        .subscribe({
          next: (response: AssistantResponse) => {
            // Remove loading message
            this.messages = this.messages.filter(m => m.type !== 'loading');
            
            // Add assistant's main text response
            if (response.answer) {
              const assistantMessage: ChatMessage = {
                text: response.answer,
                isUser: false,
                type: 'text'
              };
              this.messages.push(assistantMessage);
            }

            // Process supporting data
            if (response.supporting_data) {
              if (response.supporting_data.similar_incidents && response.supporting_data.similar_incidents.results.length > 0) {
                const similarIncidentsMessage: ChatMessage = {
                  text: 'Here are some similar incidents I found:',
                  isUser: false,
                  type: 'similar_incidents',
                  data: response.supporting_data.similar_incidents.results
                };
                this.messages.push(similarIncidentsMessage);
              }
              if (response.supporting_data.recommend_resolutions && response.supporting_data.recommend_resolutions.recommendations.length > 0) {
                const recommendationsMessage: ChatMessage = {
                  text: 'You could also try these recommended resolutions:',
                  isUser: false,
                  type: 'recommended_resolutions',
                  data: response.supporting_data.recommend_resolutions.recommendations
                };
                this.messages.push(recommendationsMessage);
              }
            }
            
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
            this.messages = this.messages.filter(m => m.type !== 'loading');
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
    // Don't save loading messages, type 'loading' will be filtered
    const messagesToSave = this.messages.filter(m => m.type !== 'loading');
    localStorage.setItem('chatMessages', JSON.stringify(messagesToSave));
  }

  clearConversation(): void {
    this.messages = [];
    this.sessionId = null;
    localStorage.removeItem('chatSessionId');
    localStorage.removeItem('chatMessages');
  }
}
