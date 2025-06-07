import { Component, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrls: ['./app.scss'],
})
export class AppComponent implements AfterViewChecked {
  userInput = '';
  loading = false;
  messages: { text: string; type: 'user' | 'bot' }[] = [];

  @ViewChild('chatBox') chatBox!: ElementRef;

  sendMessage() {
    const question = this.userInput.trim();
    if (!question) return;

    this.messages.push({ text: question, type: 'user' });
    this.userInput = '';
    this.loading = true;

    fetch('http://localhost:8000/ask-assistant', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    })
      .then(res => res.json())
      .then(data => {
        this.messages.push({ text: data.answer || 'No answer returned.', type: 'bot' });
      })
      .catch(() => {
        this.messages.push({ text: 'Failed to connect to backend.', type: 'bot' });
      })
      .finally(() => {
        this.loading = false;
      });
  }

  ngAfterViewChecked() {
    if (this.chatBox) {
      this.chatBox.nativeElement.scrollTop = this.chatBox.nativeElement.scrollHeight;
    }
  }
}

