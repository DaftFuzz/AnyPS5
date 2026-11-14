#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace AgcDriver::Kyty {

template <typename Notice>
class NestedChanges {
public:
    bool Open() const {
        return depth != 0;
    }

    template <typename Lock>
    void Begin(Lock&& lock) {
        if (depth++ == 0) lock();
    }

    template <typename Unlock, typename Send>
    void End(Unlock&& unlock, Send&& send) {
        if (--depth != 0) return;
        unlock();
        while (!held.empty()) {
            auto batch = std::move(held);
            held.clear();
            for (const auto& notice : batch) send(notice);
        }
    }

    template <typename Send>
    void Notify(bool deferNested, const Notice& notice, Send&& send) {
        if (deferNested && depth != 0) {
            held.push_back(notice);
            return;
        }
        send(notice);
    }

    std::size_t Held() const {
        return held.size();
    }

private:
    int depth = 0;
    std::vector<Notice> held;
};

}
