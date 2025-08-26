var test = false;

// set image as background when uploaded
var src = document.getElementById("car-image");
var target = document.getElementById("custom-upload")
var constraints = { audio: false, video: { width: document.getElementById("video-feed-main").offsetWidth, height:document.getElementById("video-feed-main").offsetHeight} };


function showImage(src,target) {
    var fr=new FileReader();
    // when image is loaded, set the src of the image where you want to display it
    fr.onload = function(e) { target.style.backgroundImage = `url(${ this.result })`; };
    src.addEventListener("change",function() {
      // fill fr with image data    
      fr.readAsDataURL(src.files[0]);
    });
  }
showImage(src,target);

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

