document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('.entry');
  
    form.addEventListener('submit', function(event) {
      event.preventDefault(); // Prevent default form submission
  
      const disasterType = document.getElementById('disasterType').value;
      const magnitude = document.getElementById('magnitude').value;
  
      // Create the next page's filename based on the disaster type
      const nextPageFilename = `${disasterType.toLowerCase().replace(/\s+/g, '-')}.html`;
  
      // Store data in localStorage (or sessionStorage)
      localStorage.setItem('disasterData', JSON.stringify({
        disasterType: disasterType,
        magnitude: magnitude,
        population: 21600,
        infrastructureCost: 84.247,
      }));
  
      // Redirect to the next page
      window.location.href = nextPageFilename;
    });
  });