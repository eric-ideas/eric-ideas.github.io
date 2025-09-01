// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "A growing collection of your cool projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-gallery",
          title: "gallery",
          description: "A growing collection of my gallery works.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/gallery/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "This is a description of the page. You can modify it in &#39;_pages/cv.md&#39;. You can also change or remove the top pdf download button.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "nav-teaching",
          title: "Teaching",
          description: "Courses, lectures, and materials I have taught.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/teaching/";
          },
        },{id: "post-linear-algebra-ii-2025-notes",
        
          title: "Linear Algebra II 2025 Notes",
        
        description: "Lecture notes and materials",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/linear-algebra-ii-2025/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "gallery-spectral-theorem-notes",
          title: 'Spectral Theorem Notes',
          description: "",
          section: "Gallery",handler: () => {
              window.location.href = "/gallery/example-g/";
            },},{id: "news-i-am-currently-taking-mged11-stab52-matb41-matb44-stab40-cscc37-in-order-theory-and-practice-of-regression-analysis-an-introduction-to-probability-mathematical-treatment-techniques-of-the-calculus-of-several-variables-i-differential-equation-i-fundemental-of-investment-and-credit-and-introduction-to-numerical-methods-see-my-blog-named-introspection-for-my-academic-journal-i-update-what-i-learned-and-some-of-my-remarks-or-thoughts-each-week",
          title: 'I am currently taking MGED11, STAB52, MATB41, MATB44, STAB40, CSCC37. In order: Theory...',
          description: "",
          section: "News",},{id: "projects-intermediate-microeconomics",
          title: 'Intermediate Microeconomics',
          description: "",
          section: "Projects",handler: () => {
              window.location.href = "/projects/empty/";
            },},{id: "teaching-intermediate-microeconomics",
          title: 'Intermediate Microeconomics',
          description: "",
          section: "Teaching",handler: () => {
              window.location.href = "/teaching/teaching_ex/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%6F%75@%65%78%61%6D%70%6C%65.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-inspire',
        title: 'Inspire HEP',
        section: 'Socials',
        handler: () => {
          window.open("https://inspirehep.net/authors/1010907", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=qc6CJjYAAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.alberteinstein.com/", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
