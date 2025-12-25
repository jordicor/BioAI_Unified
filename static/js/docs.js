// API Documentation Interactive Features

document.addEventListener('DOMContentLoaded', function() {
    // Initialize sidebar navigation
    initSidebarNavigation();
    
    // Initialize code copy functionality
    initCodeCopyButtons();
    
    // Initialize example tabs
    initExampleTabs();
    
    // Initialize smooth scrolling
    initSmoothScrolling();
});

function initSidebarNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.docs-section');
    
    // Highlight active section in navigation
    function updateActiveNav() {
        let currentSection = '';
        
        sections.forEach(section => {
            const rect = section.getBoundingClientRect();
            if (rect.top <= 100 && rect.bottom >= 100) {
                currentSection = section.id;
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${currentSection}`) {
                link.classList.add('active');
            }
        });
    }
    
    // Update on scroll
    window.addEventListener('scroll', updateActiveNav);
    updateActiveNav(); // Initial update
}

function initCodeCopyButtons() {
    const copyButtons = document.querySelectorAll('.copy-btn');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const codeBlock = this.closest('.code-example').querySelector('code');
            const text = codeBlock.textContent;
            
            navigator.clipboard.writeText(text).then(() => {
                const originalText = this.textContent;
                this.textContent = 'Copied!';
                this.style.background = '#10b981';
                this.style.color = 'white';
                this.style.borderColor = '#10b981';
                
                setTimeout(() => {
                    this.textContent = originalText;
                    this.style.background = '';
                    this.style.color = '';
                    this.style.borderColor = '';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
                this.textContent = 'Failed';
                this.style.background = '#ef4444';
                this.style.color = 'white';
                this.style.borderColor = '#ef4444';
                
                setTimeout(() => {
                    this.textContent = 'Copy';
                    this.style.background = '';
                    this.style.color = '';
                    this.style.borderColor = '';
                }, 2000);
            });
        });
    });
}

function initExampleTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const language = this.getAttribute('onclick').match(/'([^']+)'/)[1];
            showExample(language);
        });
    });
}

function showExample(language) {
    // Update button states
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('onclick').includes(language)) {
            btn.classList.add('active');
        }
    });
    
    // Update content visibility
    const exampleContents = document.querySelectorAll('.example-content');
    exampleContents.forEach(content => {
        content.classList.remove('active');
        if (content.id === `${language}-example`) {
            content.classList.add('active');
        }
    });
}

function initSmoothScrolling() {
    const navLinks = document.querySelectorAll('.nav-link[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Mobile sidebar toggle (for future mobile implementation)
function toggleSidebar() {
    const sidebar = document.querySelector('.docs-sidebar');
    sidebar.classList.toggle('open');
}

// Utility function for copying code blocks
function copyCode(button) {
    // This function is called by inline onclick handlers
    // The actual logic is handled by initCodeCopyButtons()
}