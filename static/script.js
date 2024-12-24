// async function submitForm() {
//     const form = document.getElementById('predictionForm');
//     const formData = new FormData(form);

//     // Convert FormData to JSON
//     const formJSON = Object.fromEntries(formData.entries());

//     try {
//         const response = await fetch('http://localhost:5000/predict', {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify(formJSON)
//         });

//         if (!response.ok) {
//             throw new Error('Error: ' + response.statusText);
//         }

//         const data = await response.json();
//         document.getElementById('results').textContent = JSON.stringify(data, null, 2);
//     } catch (error) {
//         console.error('Error:', error);
//         document.getElementById('results').textContent = `Error: ${error.message}`;
//     }
// }


const form = document.getElementById('predictionForm');

form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const formData = new FormData(form);

    // Convert FormData to JSON
    const formJSON = Object.fromEntries(formData.entries());

    console.log("Form Data",formJSON);
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formJSON)
        });

        if (!response.ok) {
            throw new Error('Error: ' + response.statusText);
        }

        const data = await response.json();
        document.getElementById('results').textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('results').textContent = `Error: ${error.message}`;
    }
});