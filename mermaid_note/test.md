flowchart LR
    %% 定义样式类
    classDef styleA fill:#F9E79F,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    classDef styleB fill:#A9DFBF,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    classDef styleC fill:#81D4FA,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    classDef styleD fill:#F8BBD0,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    classDef styleE fill:#D1C4E9,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    classDef styleF fill:#FFCCBC,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    classDef styleG fill:#C8E6C9,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    classDef styleH fill:#BBDEFB,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    classDef styleI fill:#FFECB3,stroke:#333,stroke-width:2px,rx:8px,ry:8px;

    %% 流程图节点（使用类样式）
    A[输入：源语言文本<br>（如“我爱学习”）]:::styleA -->|提供原始数据| B[预处理：源语言嵌入+位置编码<br>（转向量+标记语序）]:::styleB
    B -->|转换为特征向量| C[Encoder：提取源语言特征<br>→ 捕捉“我-爱-学习”语义关联]:::styleC
    C -->|提供语义表示| D[Decoder：生成目标语言序列<br>（初始输入为“<开始>”标记）]:::styleD
    D -->|防止信息泄露| D1[掩码注意力：只看已生成词<br>（避免“偷看”未生成的英文）]:::styleE
    D1 -->|建立语言关联| D2[跨注意力：关联Encoder的源语言特征<br>（让“love”对应“爱”的含义）]:::styleF
    D2 -->|优化特征表示| D3[前馈网络：优化目标语言特征]:::styleG
    D3 -->|生成概率分布| E[输出层：目标语言词表概率<br>（预测下一个词，如“love”）]:::styleH
    E -->|循环迭代生成| F[循环生成：将“love”接回Decoder<br>→ 直到生成“<结束>”标记]:::styleI
    F -->|完成翻译过程| G[最终输出：目标语言译文<br>（如“I love studying”）]:::styleA