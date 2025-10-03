```mermaid
flowchart LR
    %% 定义样式类
    classDef styleA fill:#F9E79F,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    classDef styleB fill:#A9DFBF,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    classDef styleC fill:#81D4FA,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    classDef styleD fill:#F8BBD0,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    
    %% 流程图节点（直接绑定样式）
    A[设计一个模型]:::styleA --> B[获取新数据]:::styleB
    B --> C[更新模型]:::styleC
    C --> D[检查是否够好]:::styleD
    D --> B
```