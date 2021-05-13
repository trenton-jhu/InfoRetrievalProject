function handleSubmit() {
    const form = document.getElementById("form-genre");
    const note = document.getElementById("genre-result");
    note.innerText = "Classifying ... Please wait!";
    const query = form[0].value;

    const request = new XMLHttpRequest();
    request.open('GET', 'http://127.0.0.1:5000/genre?data=' + query);
    request.onload = () => {
        const result = JSON.parse(request.response);
        if (request.status === 200) {
            note.style.display = 'block';
            note.innerText = "Predicted genre for your plot: " + result.genre;
        }
    };
    request.send();
}

function handleSubmit2() {
    const form = document.getElementById("form-genres");
    const note = document.getElementById("genres-result");
    note.innerText = "Classifying ... Please wait!";
    const query = form[0].value;

    const request = new XMLHttpRequest();
    request.open('GET', 'http://127.0.0.1:5000/genres?data=' + query);
    request.onload = () => {
        const result = JSON.parse(request.response);
        if (request.status === 200) {
            note.style.display = 'block';
            note.innerText = "Predicted genre for your plot: " + result.genres;
        }
    };
    request.send();
}

function handleSubmit3() {
    const form = document.getElementById("form-sa");
    const note = document.getElementById("sa-result");
    note.innerText = "Classifying ... Please wait!";
    const query = form[0].value;

    const request = new XMLHttpRequest();
    request.open('GET', 'http://127.0.0.1:5000/sent?data=' + query);
    request.onload = () => {
        const result = JSON.parse(request.response);
        if (request.status === 200) {
            note.style.display = 'block';
            note.innerText = "Predicted sentiment for your movie review: " + result.label;
        }
    };
    request.send();
}

function handleInit(type) {
    document.getElementById("init").style.display = "none";
    document.getElementById("feedback").style.display = "block";
    document.getElementById("like").onclick = () => handleFeedback(0);
    document.getElementById("dislike").onclick = () => handleFeedback(1);
    const request = new XMLHttpRequest();
    request.open('GET', 'http://127.0.0.1:5000/init?mode=' + type);
    request.onload = () => {
        const result = JSON.parse(request.response);
        if (request.status === 200) {
            displayMovies(result);
        }
    };
    request.send();
}

function handleFeedback(label) {
    const request = new XMLHttpRequest();
    request.open('GET', 'http://127.0.0.1:5000/feedback?data=' + label);
    request.onload = () => {
        const result = JSON.parse(request.response);
        if (request.status === 200) {
            displayMovies(result);
        }
    };
    request.send();
}

function displayMovies(data) {
    const note = document.getElementById("prediction");
    if (data.prediction === 0) {
        note.innerText = "We predict that you would like this movie!";
    } else {
        note.innerText = "We predict that you would not like this movie!"
    }
    document.getElementById("poster").src = data.image;
    document.getElementById("imdb").href = data.url;
    document.getElementById("title").innerText = data.title;
    document.getElementById("genre").innerText = data.genre;
    document.getElementById("director").innerText = data.director;
    document.getElementById("cast").innerText = data.cast;
    document.getElementById("runtime").innerText = data.runtime + " min";
    document.getElementById("summary").innerText = data.summary;
    document.getElementById("movie-card").style.display = "block";
}


document.getElementById("submit").onclick = handleSubmit;
document.getElementById("submit2").onclick = handleSubmit2;
document.getElementById("submit3").onclick = handleSubmit3;
document.getElementById("nearest-centroid").onclick = () => handleInit("nearest");
document.getElementById("knn").onclick = () => handleInit("knn");

