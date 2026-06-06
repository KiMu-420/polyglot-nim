package main

import (
    "log"
    "net/http"
    "net/http/httputil"
    "net/url"
)

func main() {
    target, _ := url.Parse("http://localhost:8080")
    proxy := httputil.NewSingleHostReverseProxy(target)
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        log.Printf("%s %s", r.Method, r.URL.Path)
        proxy.ServeHTTP(w, r)
    })
    log.Println("Go gateway listening on :80")
    log.Fatal(http.ListenAndServe(":80", nil))
}
