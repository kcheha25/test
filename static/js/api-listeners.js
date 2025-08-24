document.addEventListener('DOMContentLoaded', function() {
    const videoStream = document.getElementById('video-main');
    console.log(videoStream);
    videoStream.src = '/api/vfeed';
  });

document.addEventListener('DOMContentLoaded', function() {
    const eventSource = new EventSource('/api/stream'); 

    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        console.log("Data received from Flask:", data);

        const resultElement = document.getElementById('ai-result');
        const instructElement = document.getElementById('ai-instruct');
        const instructIcon = document.getElementById('ai-result-icon');
        const instructDiv = document.getElementById('ai-action-div');


        if (Array.isArray(data.new_content)) {
            if (resultElement) resultElement.textContent = data.new_content[0];
            if (instructElement) instructElement.textContent = data.new_content[1];
            
            if(data.new_content[2] == "success") {
                instructIcon.src = 'static/assets/validate.png'
                instructDiv.style.color = "#0ed145";
            } else if (data.new_content[2] == "detected") {
                instructIcon.src = 'static/assets/car_accessing.png'
                instructDiv.style.color = "#ec1c24";
            } else {
                instructIcon.src = 'static/assets/pending.png'
                instructDiv.style.color = "#ffca18";
            }
        }
    };
});