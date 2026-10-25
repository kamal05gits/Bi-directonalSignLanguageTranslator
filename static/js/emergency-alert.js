(() => {
    const button = document.getElementById('emergency-alert-btn');
    const status = document.getElementById('emergency-alert-status');
    if (!button || !status) return;

    button.addEventListener('click', async () => {
        if (!window.confirm('Send an emergency alert using the configured webhook?')) return;

        button.disabled = true;
        status.textContent = 'Sending emergency alert...';
        try {
            const response = await fetch('/api/emergency-alert', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const result = await response.json();
            if (!response.ok || !result.success) {
                throw new Error(result.error || 'Emergency alert could not be sent.');
            }
            status.textContent = result.message;
            window.alert(result.message);
        } catch (error) {
            status.textContent = error.message;
            window.alert(error.message);
        } finally {
            button.disabled = false;
        }
    });
})();
