#!/bin/bash
# 监控VLA actions导出进度

while true; do
    clear
    echo "========================================="
    echo "VLA Actions导出进度监控"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="
    echo ""

    total=0
    complete=0

    for suite in libero_spatial libero_object libero_goal libero_10; do
        suite_complete=0
        echo "[$suite]"
        for i in {0..9}; do
            vla="outputs/$suite/task_$i/vla_actions_demo0.npz"
            if [ -f "$vla" ]; then
                size=$(stat -c%s "$vla" 2>/dev/null || stat -f%z "$vla" 2>/dev/null)
                echo "  task_$i: ✅ ($(numfmt --to=iec $size 2>/dev/null || echo $size))"
                complete=$((complete + 1))
                suite_complete=$((suite_complete + 1))
            else
                echo "  task_$i: ⏳"
            fi
            total=$((total + 1))
        done
        echo "  小计: $suite_complete/10"
        echo ""
    done

    percent=$((complete * 100 / total))
    echo "========================================="
    echo "总进度: $complete/$total ($percent%)"
    echo "========================================="

    if [ $complete -eq $total ]; then
        echo ""
        echo "✅ 所有VLA actions已导出完成！"
        break
    fi

    sleep 10
done
