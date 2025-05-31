document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    fetch('/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('message').textContent = 'Login successful!';
            document.getElementById('message').style.color = 'green';
            window.location.href = '/static/pages/home.html?username=' + username; // Redirect on login
        } else {
            document.getElementById('message').textContent = 'Invalid username or password.';
            document.getElementById('message').style.color = 'red';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('message').textContent = 'An error occurred during login.';
        document.getElementById('message').style.color = 'red';
    });
});

document.getElementById('toggleForm').addEventListener('click', function() {
    const isLogin = document.getElementById('loginForm').style.display !== 'none';
    document.getElementById('loginForm').style.display = isLogin ? 'none' : 'block';
    document.getElementById('registerForm').style.display = isLogin ? 'block' : 'none';
    this.textContent = isLogin ? 'Already have an account? Login' : 'Don\'t have an account? Register';
});

document.getElementById('registerForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const username = document.getElementById('regUsername').value;
    const password = document.getElementById('regPassword').value;

    fetch('/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('message').textContent = 'Registration successful! Please login.';
            document.getElementById('message').style.color = 'green';
            window.location.href = '/static/pages/home.html?username=' + username; // Redirect on registration
        } else {
            document.getElementById('message').textContent = 'Username already exists.';
            document.getElementById('message').style.color = 'red';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('message').textContent = 'An error occurred during registration.';
        document.getElementById('message').style.color = 'red';
    });
});
