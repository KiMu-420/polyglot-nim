package com.nim;

import org.springframework.web.bind.annotation.*;
import org.springframework.http.*;
import org.springframework.web.client.RestTemplate;

@RestController
@CrossOrigin(origins = "*")
public class NimController {

    private final RestTemplate restTemplate = new RestTemplate();
    private final String pythonServiceUrl = "http://localhost:5000/ask";

    @PostMapping("/api/ask")
    public ResponseEntity<?> askQuestion(@RequestBody AskRequest request,
                                         @RequestHeader(value="X-API-Key", required=false) String apiKey) {
        if (!"secret123".equals(apiKey)) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("Invalid API key");
        }
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<AskRequest> entity = new HttpEntity<>(request, headers);
        try {
            ResponseEntity<AskResponse> response = restTemplate.postForEntity(
                pythonServiceUrl, entity, AskResponse.class);
            return ResponseEntity.ok(response.getBody());
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                                 .body("Python service unreachable");
        }
    }

    static class AskRequest {
        private String question;
        public String getQuestion() { return question; }
        public void setQuestion(String question) { this.question = question; }
    }
    static class AskResponse {
        private String answer;
        public String getAnswer() { return answer; }
        public void setAnswer(String answer) { this.answer = answer; }
    }
}
