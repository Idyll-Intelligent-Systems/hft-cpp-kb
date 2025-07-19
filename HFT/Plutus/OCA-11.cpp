#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <climits>
using namespace std;

const int MAXN = 1e5 + 5;
int parent[MAXN], sz[MAXN];
bool is_special[MAXN];

int find(int u) {
    if (parent[u] == u) return u;
    return parent[u] = find(parent[u]);
}

void unite(int u, int v) {
    u = find(u);
    v = find(v);
    if (u == v) return;
    if (sz[u] < sz[v]) swap(u, v);
    sz[u] += sz[v];
    is_special[u] |= is_special[v];
    parent[v] = u;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N, M, K;
    cin >> N >> M >> K;

    vector<int> special(K);
    for (int i = 0; i < K; i++) {
        cin >> special[i];
        special[i]--; // 0-index
    }

    for (int i = 0; i < N; i++) {
        parent[i] = i;
        sz[i] = 1;
        is_special[i] = false;
    }

    for (int i : special) {
        is_special[i] = true;
    }

    for (int i = 0; i < M; i++) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        unite(u, v);
    }

    // Find all component roots and categorize them
    vector<int> normal_components;
    vector<int> special_components;
    
    for (int i = 0; i < N; i++) {
        if (find(i) == i) { // This is a root of a component
            if (is_special[i]) {
                special_components.push_back(i);
            } else {
                normal_components.push_back(i);
            }
        }
    }
    
    // If no special nodes, we need to pick the largest component
    if (special_components.empty()) {
        int max_comp = 0;
        for (int i = 1; i < N; i++) {
            if (find(i) == i && sz[i] > sz[max_comp]) {
                max_comp = i;
            }
        }
        special_components.push_back(max_comp);
        // Remove it from normal components
        normal_components.erase(
            remove(normal_components.begin(), normal_components.end(), max_comp),
            normal_components.end()
        );
    }
    
    // Find the largest special component to merge others into
    int target_comp = special_components[0];
    for (int comp : special_components) {
        if (sz[comp] > sz[target_comp]) {
            target_comp = comp;
        }
    }
    
    long long total_cost = 0;
    int target_size = sz[target_comp];
    
    // Merge all normal components into the target
    for (int comp : normal_components) {
        total_cost += (long long)sz[comp] * target_size;
        target_size += sz[comp];
    }
    
    // Merge all other special components into the target
    for (int comp : special_components) {
        if (comp != target_comp) {
            total_cost += (long long)sz[comp] * target_size;
            target_size += sz[comp];
        }
    }
    
    // Calculate maximum possible edges in complete graph
    long long max_possible_edges = (long long)N * (N - 1) / 2;
    long long new_edges = max_possible_edges - M;
    
    cout << new_edges << " " << total_cost << "\n";
}