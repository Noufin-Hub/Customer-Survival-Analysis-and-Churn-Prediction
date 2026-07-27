/**
 * Interactive Behavior for Telecom Customer Churn Dashboard
 */

document.addEventListener('DOMContentLoaded', () => {
  const churnForm = document.getElementById('churnForm');
  const predictBtn = document.getElementById('predictBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnText = document.getElementById('btnText');
  const resultsDashboard = document.getElementById('results-dashboard');

  // Handle Form Submission Loading State
  if (churnForm && predictBtn) {
    churnForm.addEventListener('submit', (e) => {
      // Check HTML5 validity
      if (!churnForm.checkValidity()) {
        return;
      }

      // Disable button & toggle spinner
      predictBtn.disabled = true;
      if (btnSpinner) {
        btnSpinner.classList.remove('d-none');
      }
      if (btnText) {
        btnText.innerHTML = '<i class="bi bi-gear-wide-connected me-2"></i> Analyzing Customer Retention Data...';
      }
    });
  }

  // Smooth scroll to results dashboard if present
  if (resultsDashboard) {
    setTimeout(() => {
      resultsDashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 250);
  }
});
