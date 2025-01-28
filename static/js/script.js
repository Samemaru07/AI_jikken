const vide = document.getElementById('myVideo');

video.addEventListener('click', () => {
    if (video.puased)
    {
        video.play();
    }
    else{
        video.puased();
    }
});