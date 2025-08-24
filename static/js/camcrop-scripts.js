var test = true;

var constraints = { audio: false, video: true};
// fonction temporaire pour tester le flux vidéo
if (test) {
    navigator.mediaDevices
    .getUserMedia(constraints)
    .then(function (mediaStream) {
        display_video(mediaStream)
    })
    .catch(function (err) {
        console.log(err.name + ": " + err.message);
    }); // always check for errors at the end.
}


function display_video(mediaStream) {
    var video = document.querySelector("video");
    video.srcObject = mediaStream;
    video.onloadedmetadata = function (e) {
      video.play();
    };
}