```mermaid
graph LR
    A[定义组件类] --> B[继承抽象基类<br>（BaseModel/BaseDataLoader）]
    B --> C[实现强制接口<br>（_build_model/_load_dataset等）]
    C --> D[用@register装饰器注册<br>（指定task_type和component_name）]
    D --> E[组件存入注册中心<br>（{task_type: {name: 类}}）]
    
    F[加载配置文件] --> G[解析task_type和component_name]
    G --> H[调用registry.get()<br>（按task+name获取组件类）]
    H --> I[传入config实例化组件<br>（model/dataloader等）]
    I --> J[拼接训练流程并执行]
```